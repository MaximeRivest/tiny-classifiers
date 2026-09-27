//#endregion
//#region src/pprint.d.ts
/** General class for pretty-printing expressions with indentation. */
declare class PPrint {
  readonly indents: number[];
  readonly lines: string[];
  constructor(indents: number[], lines: string[]);
  /** Add a fixed amount of indentation to each line. */
  indent(spaces: number): PPrint;
  /** Concatenate pretty-printed expressions with newlines. */
  concat(...items: PPrint[]): PPrint;
  /** Stack one block to the right of another one, sharing 1 common line. */
  stack(other: PPrint): PPrint;
  /** Combine this block of lines into a formatted string. */
  toString(): string;
  static pp(s: Stringable): PPrint;
}
interface Stringable {
  toString(): string;
}
//#endregion
//#region src/shape.d.ts
/** @inline */
type Pair = [number, number];
/**
 * A multidimensional view into memory. An array can be thought of as the
 * combination of a linear buffer of memory, along with a `View`.
 *
 * Formula for getting a data point is basically:
 *   1. Check if ∀i. 0 <= dim[i] < shape[i], otherwise out of bounds.
 *   2. If mask exists, and ∃i. dim[i] ∉ mask[i], return 0.
 *   2. Otherwise, look at this memory address: offset + ∑(strides[i] * dim[i]).
 */
declare class View {
  #private;
  /** The shape of the view (size of each dimension). */
  readonly shape: number[];
  /** How many indices to move in buffer for each hop in one dimension. */
  readonly strides: number[];
  /** Offset from the start of the buffer. */
  readonly offset: number;
  /** Masked out subarray where data is read. All other data is zeroed. */
  readonly mask: Pair[] | null;
  private constructor();
  static create(shape: number[], strides?: number[], offset?: number, mask?: Pair[] | null): View;
  get ndim(): number;
  get size(): number;
  /** Whether this is a default, contiguous, unaltered view of the data (identity). */
  get contiguous(): boolean;
  /** Return the range of data being indexed in this view, or [0, 0] if none. */
  dataRange(): [number, number];
  /** Produce an AluExp for evaluating this view at an index. */
  toAluExp(idxs: AluExp[]): [AluExp, AluExp];
  /**
   * Try to compose this view with another one. `this` view is applied first,
   * followed by the argument. If this is not possible for the specific views,
   * return `null` instead.
   *
   * If composable, return a combined view with the same shape as `v1`.
   *
   * This is very tricky. The shapes of v1 and v2 may be different, and in that
   * case, we do some math to figure out whether they're compatible.
   */
  compose(v1: View): View | null;
  /** Attempt to simplify this view into a smaller reshaped form. */
  minify(): View;
  /** Pad the view with zeros on each dimension. */
  pad(arg: Pair[]): View;
  /** Shrink the view by taking a subarray. */
  shrink(arg: Pair[]): View;
  /** Expand one or more axes with length "1" by repeating the data. */
  expand(newShape: number[]): View;
  /** Permute the axes of an array. */
  permute(axis: number[]): View;
  /** Flip (reverse) one or more axes of the view. */
  flip(arg: boolean[]): View;
  /** Reshape the view into a new shape. */
  reshape(newShape: number[]): View | null;
}
/**
 * Array shape after applying movement operations, as a series of views.
 *
 * Each view is applied, then treated as if it were a contiguous array of its
 * shape, then used as the virtual buffer for the next view.
 */
declare class ShapeTracker {
  readonly views: View[];
  constructor(views: View[]);
  /** Compose this shape tracker with another, applying it after this one. */
  compose(other: ShapeTracker): ShapeTracker;
  static fromShape(shape: number[]): ShapeTracker;
  get contiguous(): boolean;
  get consecutive(): boolean;
  get lastStrides(): number[];
  get shape(): number[];
  get size(): number;
  toAluExp(idxs: AluExp[]): [AluExp, AluExp];
  simplify(): ShapeTracker;
  pad(arg: Pair[]): ShapeTracker;
  shrink(arg: Pair[]): ShapeTracker;
  expand(newShape: number[]): ShapeTracker;
  permute(axis: number[]): ShapeTracker;
  flip(arg: boolean[]): ShapeTracker;
  reshape(newShape: number[]): ShapeTracker;
  /** Broadcast along the given new axes, then expand the shape. */
  broadcast(newShape: number[], axis: number[]): ShapeTracker;
  /**
   * Repeat data in each axis by a positive number of repetitions.
   *
   * - If `tile` is true (default): [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
   * - If `tile` is false: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].
   */
  repeat(reps: number[], tile?: boolean): ShapeTracker;
  /** Move axis i to axis j. */
  moveaxis(i: number, j: number): ShapeTracker;
  /** Like pad(), but allows for negative values. */
  padOrShrink(arg: Pair[]): ShapeTracker;
}
//#endregion
//#region src/utils.d.ts
/**
 * Set the debug level for verbose logging.
 *
 * 1. JIT compile logs
 * 2. Shader code
 * 3. Expressions and metadata
 * 4. JIT programs, tuning details
 * 5. Most verbose operation traces
 *
 * This is an experimental API and may change in behavior. Do not rely on this
 * in production.
 */
declare function setDebug(level: number): void;
/** @inline */
type RecursiveArray<T> = T | RecursiveArray<T>[];
interface FpHashable {
  hash(state: FpHash): void;
}
/**
 * Polynomial hashes modulo p are good at avoiding collisions in expectation.
 * Probability-wise, it's good enough to be used for something like
 * deduplicating seen compiler expressions, although it's not adversarial.
 *
 * See https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(number_theory)
 */
declare class FpHash {
  #private;
  value: bigint;
  update(x: string | boolean | number | bigint | null | undefined | FpHashable): this;
  static hash(...values: (string | boolean | number | bigint | null | undefined | FpHashable)[]): bigint;
}
//#endregion
//#region src/alu.d.ts
/** A numerical data type for array contents. */
declare enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Uint32 = "uint32",
  Bool = "bool",
  Float16 = "float16",
  Float64 = "float64"
}
/** @inline */
type DataArray = Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer> | Float16Array<ArrayBuffer> | Float64Array<ArrayBuffer>;
/**
 * Promote two dtypes to their join according to the type lattice.
 *
 * When performing operations between arrays of different types, we need to
 * promote both operands to a common type that can represent values from both
 * input types. This follows JAX's type promotion rules.
 *
 * **Type lattice:**
 * ```text
 * bool -> uint32 -> int32 -> float16 -> float32 -> float64
 *  weakType --^
 * ```
 *
 * `weakType` represents weakly typed arrays. These are created for JS numbers,
 * which default to float32 but "weak" so they cast to the dtype of any array
 * they are first combined with, except `bool`.
 *
 * **Examples:**
 * - `promoteTypes(bool, int32) → int32`
 * - `promoteTypes(uint32, int32) → int32`
 * - `promoteTypes(int32, float16) → float16`
 * - `promoteTypes(float16, float32) → float32`
 * - `promoteTypes(uint32, float32) → float32`
 */
declare function promoteTypes(dtype1: DType, dtype2: DType): DType;
/**
 * Mathematical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
declare class AluExp implements FpHashable {
  #private;
  readonly op: AluOp;
  readonly dtype: DType;
  readonly src: AluExp[];
  readonly arg: any;
  constructor(op: AluOp, dtype: DType, src: AluExp[], arg?: any);
  static add(a: AluExp, b: AluExp): AluExp;
  static sub(a: AluExp, b: AluExp): AluExp;
  static mul(a: AluExp, b: AluExp): AluExp;
  static idiv(a: AluExp, b: AluExp): AluExp;
  static mod(a: AluExp, b: AluExp): AluExp;
  static min(a: AluExp, b: AluExp): AluExp;
  static max(a: AluExp, b: AluExp): AluExp;
  static bitCombine(a: AluExp, b: AluExp, mode: "and" | "or" | "xor"): AluExp;
  static bitShift(a: AluExp, b: AluExp, mode: "shl" | "shr"): AluExp;
  static bitCount(a: AluExp): AluExp;
  static sin(a: AluExp): AluExp;
  static cos(a: AluExp): AluExp;
  static asin(a: AluExp): AluExp;
  static atan(a: AluExp): AluExp;
  static exp(a: AluExp): AluExp;
  static log(a: AluExp): AluExp;
  static erf(a: AluExp): AluExp;
  static erfc(a: AluExp): AluExp;
  static sqrt(a: AluExp): AluExp;
  static floor(a: AluExp): AluExp;
  static ceil(a: AluExp): AluExp;
  static reciprocal(a: AluExp): AluExp;
  static cast(dtype: DType, a: AluExp): AluExp;
  static bitcast(dtype: DType, a: AluExp): AluExp;
  static threefry2x32(k0: AluExp, k1: AluExp, c0: AluExp, c1: AluExp, mode?: "xor" | 0 | 1): AluExp;
  static cmplt(a: AluExp, b: AluExp): AluExp;
  static cmpne(a: AluExp, b: AluExp): AluExp;
  static where(cond: AluExp, a: AluExp, b: AluExp): AluExp;
  static const(dtype: DType, value: any): AluExp;
  static special(dtype: DType, name: string, n: number): AluExp;
  static variable(dtype: DType, name: string): AluExp;
  static globalIndex(dtype: DType, gid: number, len: number, bufidx: AluExp): AluExp;
  static globalView(dtype: DType, gid: number, st: ShapeTracker, indices: AluExp[]): AluExp;
  static f32(value: number): AluExp;
  static i32(value: number): AluExp;
  static u32(value: number): AluExp;
  static bool(value: boolean): AluExp;
  static f16(value: number): AluExp;
  static f64(value: number): AluExp;
  not(): AluExp;
  /** Compute a reasonable expression hash with low collision rate. */
  getHash(): bigint;
  hash(state: FpHash): void;
  /** Substitute variables in this AluExp to values. */
  substitute(variables: Record<string, AluExp>): AluExp;
  /** Reindex gid values in this expression as needed. */
  reindexGids(newGids: number[]): AluExp;
  get min(): number;
  get max(): number;
  /** Largest known integer that divides self. */
  constFactor(): number;
  /**
   * Checks if divisible by an integer v and returns the quotient if it is, or
   * `null` if it's not divisible.
   */
  divides(v: number): AluExp | null;
  /**
   * Get all expressions by deeply matching an operation.
   *
   * For example: `((2+(3*5))+4).splitOp(+) -> [2,(3*5),4]`.
   */
  splitOp(sep: AluOp): IterableIterator<AluExp>;
  /**
   * Simplify the expression by replacing any known patterns and deduping
   * identical subexpressions.
   */
  simplify(cache?: Map<bigint, AluExp>): AluExp;
  /** Resolve this to a value, or `undefined` if not possible. */
  resolve(): any | undefined;
  /**
   * Evaluate the expression on CPU, returning the result.
   *
   * Typically you would compile the AluExp as a representation to a lower-level
   * language. This is just to define the semantics and help debug.
   *
   * Note that the representation of Bool is as a number (0 or 1) here.
   */
  evaluate(context: Record<string, any>, globals?: (gid: number, bufidx: number) => any): number;
  /** Get this expression in debug format as a string. */
  toString(): string;
  /** Generic fold() operation with a reducer over the expression tree. */
  fold<T = void>(reducer: (exp: AluExp, mappedSrc: T[]) => T): T;
  /** Check if any expression in the tree satisfies a predicate. */
  some(predicate: (exp: AluExp) => boolean): boolean;
  /** Rewrite the expression recursively using a visitor. */
  rewrite(visitor: (exp: AluExp) => AluExp | undefined | null): AluExp;
  /** Collect all nodes that satisfy a predicate. */
  collect(predicate: (exp: AluExp) => boolean): AluExp[];
  /** Produce all distinct AluOp in this expression, with their dtypes. */
  distinctOps(): Map<AluOp, Set<DType>>;
  /** Rewrite GlobalView operations to GlobalIndex operations. */
  rewriteGlobalViews(): AluExp;
}
/** Symbolic form for each mathematical operation. */
declare enum AluOp {
  Add = "Add",
  Sub = "Sub",
  Mul = "Mul",
  Idiv = "Idiv",
  Mod = "Mod",
  Min = "Min",
  Max = "Max",
  BitCombine = "BitCombine",
  // arg = 'or' | 'and' | 'xor'
  BitInvert = "BitInvert",
  BitShift = "BitShift",
  // arg = 'shl' | 'shr'
  BitCount = "BitCount",
  Sin = "Sin",
  Cos = "Cos",
  Asin = "Asin",
  Atan = "Atan",
  Exp = "Exp",
  Log = "Log",
  Erf = "Erf",
  Erfc = "Erfc",
  Sqrt = "Sqrt",
  Floor = "Floor",
  Ceil = "Ceil",
  Reciprocal = "Reciprocal",
  Cast = "Cast",
  Bitcast = "Bitcast",
  Cmplt = "Cmplt",
  Cmpne = "Cmpne",
  Where = "Where",
  // Ternary operator: `cond ? a : b`
  Threefry2x32 = "Threefry2x32",
  // PRNG operation, arg = 'xor' | 0 | 1
  Const = "Const",
  // arg = value
  Special = "Special",
  // arg = [variable, n]
  Variable = "Variable",
  // arg = variable
  GlobalIndex = "GlobalIndex",
  // arg = [gid, len]; src = [bufidx]
  GlobalView = "GlobalView"
}
/**
 * Description of a kernel to be compiled.
 *
 * Each of these can be processed by a backend into some lower-level
 * representation. It consists of one or more fused operations, optionally
 * indexing into a buffer.
 */
declare class Kernel implements FpHashable {
  /** Number of global arguments / arrays. */
  readonly nargs: number;
  /** Size of the result array in element count. */
  readonly size: number;
  /** Expression to be evaluated. */
  readonly exp: AluExp;
  /** Optional reduction to be performed. */
  readonly reduction?: Reduction | undefined;
  constructor(/** Number of global arguments / arrays. */

  nargs: number, /** Size of the result array in element count. */

  size: number, /** Expression to be evaluated. */

  exp: AluExp, /** Optional reduction to be performed. */

  reduction?: Reduction | undefined);
  hash(state: FpHash): void;
  pprint(): PPrint;
  toString(): string;
  /** The dtype of the values output by this kernel. */
  get dtype(): DType;
  /** The number of bytes in the output array when evaluating this kernel. */
  get bytes(): number;
}
/**
 * Description of a reduction.
 *
 * The strategy of jax-js backends is to either handle a standard operation that
 * is dispatched in a vectorized way over an array, or to reduce over one axis
 * of some computation. This is a description of the reduction.
 *
 * Reduction only supports a few operations, and only over one axis. Users can
 * always `flatten()` the array before reducing if needed.
 *
 * The backend is responsible for implementing the reduction in a way that
 * minimizes the number of global memory loads, for efficiency. This involves
 * passing through some optimization strategy. But optimizations are not coded
 * at this level since they depend on GPU, versus CPU or Wasm.
 */
declare class Reduction implements FpHashable {
  /** Data type of the values being reduced over. */
  readonly dtype: DType;
  /** Operation to perform. Only ops in `AluGroup.Reduce` are supported. */
  readonly op: AluOp;
  /** Size of the reduction axis. */
  readonly size: number;
  /** Follow-up expression defined with the "acc" variable, defaults to identity. */
  readonly epilogue: AluExp;
  constructor(/** Data type of the values being reduced over. */

  dtype: DType, /** Operation to perform. Only ops in `AluGroup.Reduce` are supported. */

  op: AluOp, /** Size of the reduction axis. */

  size: number, /** Follow-up expression defined with the "acc" variable, defaults to identity. */

  epilogue?: AluExp);
  hash(state: FpHash): void;
  toString(): string;
  /** Get the identity for this reduction operation. */
  get identity(): any;
  /** Evaluate this operation on CPU. */
  evaluate(...values: any): any;
}
//#endregion
//#region src/routine.d.ts
/**
 * Advanced operations that don't fit into the `AluExp` compiler representation.
 *
 * Some routines like iterative matrix algorithms, FFTs, or sorting may not be
 * easy to express efficiently as a `Kernel` object. These also tend to be
 * somewhat expensive, so the benefit of kernel fusion and inlining is less
 * relevant.
 *
 * For these operations, we dispatch them as a custom operation on the backend,
 * which each backend implements in a specific way. These are listed in the
 * `Routines` enum below.
 *
 * Routines cannot be fused into other kernels and always operate on contiguous
 * arrays (default `ShapeTracker`).
 */
declare class Routine {
  /** The name of the routine. */
  readonly name: Routines;
  /** Dtype and shape of the inputs and outputs. */
  readonly type: RoutineType;
  /** Extra parameters specific to the routine. */
  readonly params?: any | undefined;
  constructor(/** The name of the routine. */

  name: Routines, /** Dtype and shape of the inputs and outputs. */

  type: RoutineType, /** Extra parameters specific to the routine. */

  params?: any | undefined);
}
/** One of the valid `Routine` that can be dispatched to backend. */
declare enum Routines {
  /**
   * Sort along the last axis.
   *
   * This may be _unstable_ but it often doesn't matter, sorting numbers is
   * bitwise unique up to signed zeros and NaNs.
   */
  Sort = "Sort",
  /** Stable sorting, returns `int32` indices and values of the sorted array. */
  Argsort = "Argsort",
  /**
   * Scatter updates into a zero-filled output, either replacing or adding at
   * each destination according to `ScatterOp`.
   */
  Scatter = "Scatter",
  /**
   * Solve a triangular system of equations.
   *
   * The first batch of inputs `A` should be of shape `[..., N, N]` and upper
   * triangular, while the second batch `B` should be of shape `[..., M, N]`.
   *
   * Solves for `X` in the equation `A @ X.T = B.T`, where `A` is the
   * triangular matrix. This is equivalent to `X = B @ A^-T`.
   */
  TriangularSolve = "TriangularSolve",
  /**
   * Cholesky decomposition of 2D positive semi-definite matrices.
   *
   * The input batch should be of shape `[..., N, N]`, and the output batch is
   * of the same shape, containing the lower-triangular matrix `L` such that
   * `A = L @ L.T`. Behavior is unspecified if A is not positive semi-definite.
   */
  Cholesky = "Cholesky",
  /**
   * LU decomposition of 2D rectangular matrices.
   *
   * The input is a batch of shape `[..., M, N]`, and the output is a tuple of
   * three arrays: `LU, Pivots, Permutation`.
   *
   * - `LU` is of shape `[..., M, N]`, containing the combined lower and upper
   *   triangular matrices. (lower triangular = implicit unit diagonal)
   * - `Pivots` is of shape `[..., min(M, N)]`, containing the row swaps.
   * - `Permutation` is of shape `[..., M]`, containing the permutation vector
   *   such that `P = eye(M).slice(Permutation)` -> `P @ A = L @ U`.
   */
  LU = "LU",
  /**
   * Symmetric eigendecomposition using cyclic Jacobi rotations.
   *
   * The input is a batch of real symmetric matrices with shape `[..., N, N]`;
   * only the lower triangle is read. The output is a tuple `Diagonalized,
   * Vectors`, both shape `[..., N, N]`. `Diagonalized` is approximately
   * diagonal and `Vectors` contains the accumulated eigenvectors as columns.
   * Sorting eigenpairs is handled by the frontend wrapper.
   */
  JacobiEigh = "JacobiEigh",
  /**
   * Complex FFT along the last axis for real/imaginary array pairs.
   *
   * Inputs and outputs have identical shape, and the backend receives a
   * factorization of the FFT dimension.
   */
  Fft = "Fft"
}
/** Operation applied when scattering an update into its destination. */
declare enum ScatterOp {
  Update = "update",
  Add = "add"
}
interface ScatterParams {
  op: ScatterOp;
  shape: number[];
  axis: number[];
  outDim: number;
  uniqueIndices: boolean;
}
interface RoutineType {
  inputShapes: number[][];
  inputDtypes: DType[];
  outputShapes: number[][];
  outputDtypes: DType[];
}
//#endregion
//#region src/backend.d.ts
type Device = "cpu" | "wasm" | "webgpu" | "webgl";
declare const devices: Device[];
/** Configure the default device for arrays. */
declare function defaultDevice(device?: Device): Device;
/**
 * Initialize `jax-js` library backends.
 *
 * By default, this will initialize all available backends. If one or more
 * backends is provided, only attempt to initialize those. Returns a list of
 * available backends.
 */
declare function init(...devicesToInit: Device[]): Promise<Device[]>;
/** Unique identifier for an allocated, on-device buffer. */
type Slot = number;
/** A device backend. */
interface Backend {
  /** The name of the backend as a string. */
  readonly type: Device;
  /** Maximum number of arguments per dispatched kernel. */
  readonly maxArgs: number;
  /**
   * Allocate a new slot with reference count 1.
   *
   * Contents are unspecified unless initialData is provided.
   */
  malloc(size: number, initialData?: Uint8Array): Slot;
  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;
  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;
  /** Read a range of bytes from a buffer. */
  read(slot: Slot, start?: number, count?: number): Promise<Uint8Array<ArrayBuffer>>;
  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): Uint8Array<ArrayBuffer>;
  /** Prepare an expression to be executed later. */
  prepareKernel(kernel: Kernel): Promise<Executable>;
  /** Prepare an expression to be executed later, blocking variant. */
  prepareKernelSync(kernel: Kernel): Executable;
  /** Prepare an advanced routine to be executed later. */
  prepareRoutine(routine: Routine): Promise<Executable>;
  /** Prepare an advanced routine to be executed later, blocking variant. */
  prepareRoutineSync(routine: Routine): Executable;
  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immediately, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   */
  dispatch(exe: Executable, inputs: Slot[], outputs: Slot[]): void;
}
declare class Executable<T = any> {
  /** The `Kernel` or `Routine` that was prepared. */
  readonly source: Kernel | Routine;
  /** Extra data specific to the backend running this executable. */
  readonly data: T;
  constructor(/** The `Kernel` or `Routine` that was prepared. */

  source: Kernel | Routine, /** Extra data specific to the backend running this executable. */

  data: T);
}
/**
 * If the WebGPU backend has been initialized, return the `GPUDevice` that this
 * backend runs on. This is useful for sharing buffers.
 */
declare function getWebGPUDevice(): GPUDevice;
declare namespace tree_d_exports {
  export { JsTree, JsTreeDef, MapJsTree, NodeType, dispose, flatten, leaves, map, ref, structure, unflatten };
}
declare enum NodeType {
  Array = "Array",
  Object = "Object",
  Leaf = "Leaf"
}
/** Analog to the JAX "pytree" object, but for JavaScript. */
type JsTree<T> = T | JsTree<T>[] | {
  [key: string]: JsTree<T>;
};
type Same<X, Y> = (<T>() => T extends X ? 1 : 2) extends (<T>() => T extends Y ? 1 : 2) ? true : false;
type MappedJsTree<T, A, B> = T extends A ? B : T extends Array ? T : T extends globalThis.Array<infer U> ? number extends T["length"] ? MapJsTree<U, A, B>[] : { [K in keyof T]: MapJsTree<T[K], A, B> } : { [K in keyof T]: MapJsTree<T[K], A, B> };
/** @ignore Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure. */
type MapJsTree<T, A, B> = Same<A, B> extends true ? T : MappedJsTree<T, A, B>;
/** Represents the structure of a JsTree. */
declare class JsTreeDef {
  readonly nodeType: NodeType;
  readonly nodeMetadata: any;
  readonly childTreedefs: JsTreeDef[];
  static leaf: JsTreeDef;
  constructor(nodeType: NodeType, nodeMetadata: any, // Must be comparable with deepEqual.
  childTreedefs: JsTreeDef[]);
  /** Get the total number of leaves in the tree. */
  get size(): number;
  /** Returns a string representation of this tree definition. */
  toString(root?: boolean): string;
  /** Compare this tree definition with another. */
  equals(other: JsTreeDef): boolean;
}
/** Flatten a structured object, returning the tree definition. */
declare function flatten<T>(tree: JsTree<T>): [T[], JsTreeDef];
/** Get the leaves of a tree. */
declare function leaves<T>(tree: JsTree<T>): T[];
/** Get the treedef for a tree. */
declare function structure<T>(tree: JsTree<T>): JsTreeDef;
/** Reconstruct a structured object from the flattened representation. */
declare function unflatten<T>(treedef: JsTreeDef, leaves: Iterable<T>): JsTree<T>;
/** Maps a multi-input function over pytree args to produce a new pytree. */
declare function map<T, U, Tree extends JsTree<T>>(fn: (...args: T[]) => U, tree: Tree, ...rest: Tree[]): MapJsTree<Tree, T, U>;
/** Take a reference of every array in a tree. */
declare function ref<Tree extends JsTree<any>>(tree: Tree): Tree;
/** Dispose every array in a tree. */
declare function dispose<Tree extends JsTree<any>>(tree: Tree | null | undefined): void;
//#endregion
//#region src/frontend/convolution.d.ts
/** Definition of a general dilated convolution. Should be valid on creation. */
interface ConvParams {
  vmapDims: number;
  strides: number[];
  padding: Pair[];
  lhsDilation: number[];
  rhsDilation: number[];
}
//#endregion
//#region src/frontend/jaxpr.d.ts
/**
 * Function callback with an associated dispose() method.
 *
 * The dispose() method should be called to clean up any tracer resources needed
 * by the function after the last time it is called.
 */
type OwnedFunction<F extends Function> = F & {
  dispose: () => void;
};
/** Variable in a Jaxpr expression. */
declare class Var {
  #private;
  readonly id: number;
  readonly aval: ShapedArray;
  constructor(aval: ShapedArray);
  toString(): string;
}
/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
declare class Lit {
  readonly value: number;
  readonly aval: ShapedArray;
  get dtype(): DType;
  constructor(aval: AbstractValue, value: number);
}
type Atom = Var | Lit;
declare class VarPrinter {
  #private;
  names: Map<Var, string>;
  name(v: Var): string;
  nameType(v: Var): string;
}
/** A single statement / binding in a Jaxpr, in ANF form. */
declare class JaxprEqn {
  readonly primitive: Primitive;
  readonly inputs: Atom[];
  readonly params: Record<string, any>;
  readonly outBinders: Var[];
  constructor(primitive: Primitive, inputs: Atom[], params: Record<string, any>, outBinders: Var[]);
  pprint(usedVars?: Set<Var>, vp?: VarPrinter): PPrint;
  toString(): string;
}
/** Typed intermediate representation for traced computations. */
declare class Jaxpr implements FpHashable {
  #private;
  readonly inBinders: Var[];
  readonly eqns: JaxprEqn[];
  readonly outs: Atom[];
  constructor(inBinders: Var[], eqns: JaxprEqn[], outs: Atom[]);
  pprint(): PPrint;
  toString(): string;
  /**
   * Gets a hash of this Jaxpr.
   *
   * Var identity is not considered in the hash, so two Jaxprs with the same
   * order of assignments and operators but different variable IDs will resolve
   * to the same hash (and toString representation).
   */
  getHash(): bigint;
  hash(state: FpHash): void;
  /**
   * Produce a simplified Jaxpr with basic optimizations applied.
   *  - Trim away unused variables.
   *  - Fold away *1, *0, or +0 operations against literals.
   *  - Remove no-op movement operations.
   */
  simplify(): Jaxpr;
  /** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
  flatten(): Jaxpr;
}
/** Jaxpr with a collection of associated, traced constants. */
declare class ClosedJaxpr {
  readonly jaxpr: Jaxpr;
  readonly consts: Tracer[];
  constructor(jaxpr: Jaxpr, consts: Tracer[]);
  /** String representation of this Jaxpr. */
  toString(): string;
  /** Apply a function to the underlying Jaxpr. */
  mapJaxpr(f: (jaxpr: Jaxpr) => Jaxpr): ClosedJaxpr;
  /** Dispose of the constants in this Jaxpr. */
  dispose(): void;
}
/** @inline */
type JitOpts = {
  staticArgnums?: number[];
};
//#endregion
//#region src/frontend/core.d.ts
/**
 * Frontend primitive operations, which are lowered into Kernel objects before
 * being dispatched to the backend.
 *
 * Any operation between arrays can be described in these parts. This is also
 * the set of primitives that can occur in Jaxpr programs, and the level at
 * which transformations like vmap, grad, and jvp occur. They are loosely based
 * on [XLA](https://openxla.org/xla/operation_semantics).
 *
 * All n-ary operations support broadcasting, with NumPy semantics.
 */
declare enum Primitive {
  Add = "add",
  Mul = "mul",
  Idiv = "idiv",
  Mod = "mod",
  // uses sign of numerator, C-style, matches JS but not Python
  Min = "min",
  Max = "max",
  BitCombine = "bit_combine",
  BitShift = "bit_shift",
  BitCount = "bit_count",
  Neg = "neg",
  Reciprocal = "reciprocal",
  Floor = "floor",
  Ceil = "ceil",
  StopGradient = "stop_gradient",
  Cast = "cast",
  Bitcast = "bitcast",
  Sin = "sin",
  Cos = "cos",
  Asin = "asin",
  Atan = "atan",
  Exp = "exp",
  Log = "log",
  Erf = "erf",
  Erfc = "erfc",
  Sqrt = "sqrt",
  Reduce = "reduce",
  Dot = "dot",
  // sum(x*y, axis=-1)
  Conv = "conv",
  // see lax.conv_general_dilated
  Pool = "pool",
  PoolTranspose = "pool_transpose",
  Compare = "compare",
  Where = "where",
  Concatenate = "concatenate",
  Split = "split",
  RandomBits = "random_bits",
  Gather = "gather",
  Transpose = "transpose",
  Broadcast = "broadcast",
  Reshape = "reshape",
  Flip = "flip",
  Shrink = "shrink",
  Pad = "pad",
  Sort = "sort",
  // sort(x, axis=-1), unstable
  Argsort = "argsort",
  // argsort(x, axis=-1), stable
  Scatter = "scatter",
  TriangularSolve = "triangular_solve",
  // A is upper triangular, A @ X.T = B.T
  Cholesky = "cholesky",
  // A is positive-definite, A = L @ L^T
  LU = "lu",
  // LU decomposition with partial pivoting
  JacobiEigh = "jacobi_eigh",
  // Symmetric eigendecomposition via Jacobi rotations
  Fft = "fft",
  // Complex FFT along the final axis
  Jit = "jit"
}
interface PrimitiveParamsImpl extends Record<Primitive, Record<string, any>> {
  [Primitive.BitCombine]: {
    op: "and" | "or" | "xor";
  };
  [Primitive.BitShift]: {
    op: "shl" | "shr";
  };
  [Primitive.Cast]: {
    dtype: DType;
  };
  [Primitive.Bitcast]: {
    dtype: DType;
  };
  [Primitive.Reduce]: {
    op: AluOp;
    axis: number[];
  };
  [Primitive.Conv]: ConvParams;
  [Primitive.Pool]: {
    window: number[];
    strides: number[];
  };
  [Primitive.PoolTranspose]: {
    inShape: number[];
    window: number[];
    strides: number[];
  };
  [Primitive.Compare]: {
    op: CompareOp;
  };
  [Primitive.Concatenate]: {
    axis: number;
  };
  [Primitive.Split]: {
    axis: number;
    sizes: number[];
  };
  [Primitive.RandomBits]: {
    shape: number[];
    mode: "xor" | 0 | 1;
  };
  [Primitive.Gather]: {
    axis: number[];
    outDim: number;
    uniqueIndices: boolean;
  };
  [Primitive.Transpose]: {
    perm: number[];
  };
  [Primitive.Broadcast]: {
    shape: number[];
    axis: number[];
  };
  [Primitive.Reshape]: {
    shape: number[];
  };
  [Primitive.Flip]: {
    axis: number[];
  };
  [Primitive.Shrink]: {
    slice: Pair[];
  };
  [Primitive.Pad]: {
    width: Pair[];
  };
  [Primitive.Scatter]: ScatterParams;
  [Primitive.TriangularSolve]: {
    unitDiagonal: boolean;
  };
  [Primitive.JacobiEigh]: {
    maxSweeps: number;
    tolerance: number;
  };
  [Primitive.Fft]: {
    factors: number[];
    inverse: boolean;
  };
  [Primitive.Jit]: {
    name: string;
    jaxpr: Jaxpr;
    numConsts: number;
  };
}
/** Type of parameters taken by each primitive. */
type PrimitiveParams<T extends Primitive> = T extends keyof PrimitiveParamsImpl ? PrimitiveParamsImpl[T] : Record<string, never>;
declare enum CompareOp {
  Less = "less",
  Equal = "equal",
  NotEqual = "not_equal",
  LessEqual = "less_equal"
}
/** @inline */
type Axis = number | number[] | null;
/** @inline */
type ReduceOpts = {
  keepdims?: boolean;
};
type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace;
  globalData: any | null;
};
/**
 * Push an interpreter onto the trace stack. Use this like:
 * `using main = newMain(...);`
 */
type TracerValue = Tracer | number | boolean;
declare abstract class Trace {
  readonly main: MainTrace;
  constructor(main: MainTrace);
  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;
  abstract processPrimitive<P extends Primitive>(primitive: P, tracers: Tracer[], params: PrimitiveParams<P>): Tracer[];
}
/** Internal representation of an array value. */
interface AbstractValue {
  /** Shape of the array. Must be a static tuple of non-negative dimensions. */
  shape: number[];
  /** Concrete data type of array elements. */
  dtype: DType;
  /**
   * Arrays created from JavaScript numbers (e.g., `np.array(3)`) are created as
   * _weakly typed_ unless a dtype is explicitly specified.
   *
   * Weakly typed values will automatically cast to the data type of other
   * arrays when used as an operand as an expression. This property only affects
   * how they promote in type casting; their memory layout is still determined
   * by the actual `dtype` field.
   *
   * ```ts
   * const x = np.array(3); // weakType = true, dtype = float32
   * const y = np.array([1, 2], { dtype: np.int32 }); // weakType = false, dtype = int32
   * const z = x.add(y); // z has dtype int32 because x is weakly typed
   * ```
   *
   * Weak types are present in JIT programs in their spec (e.g., Jaxpr inputs
   * and outputs can be weakly typed) form. But they're solely a frontend
   * concept. Backends are not aware of weak types.
   */
  weakType: boolean;
}
declare abstract class Tracer {
  /** @ignore */
  readonly _trace: Trace;
  constructor(trace: Trace);
  abstract get aval(): AbstractValue;
  abstract toString(): string;
  /**
   * Access an array by reference, incrementing the reference count.
   *
   * jax-js handles freeing arrays by using "move" semantics, like in Rust/C++.
   * Whenever you pass an array into a function, that function should consume
   * the array, and it will no longer be usable. For example, if you had:
   *
   * ```
   * const x = np.array([1, 2, 3]);
   * const y = np.add(x, x);
   * ```
   *
   * The second line does not work because the first parameter consumes `x`, and
   * then the second parameter will already have been freed / disposed.
   *
   * To fix this, you can write:
   *
   * ```
   * const y = np.add(x.ref, x);
   * ```
   *
   * Under the hood, every access to `.ref` increments the internal reference
   * count of the array. The reference count starts at 1. When it hits 0, the
   * memory behind the array is freed.
   */
  abstract get ref(): this;
  /**
   * Manually decrement the reference count of the array.
   *
   * Arrays are created with reference count 1. Whenever it is used as argument
   * to a function or other operation, it is disposed (i.e., reference count
   * decreases by 1) automatically. Whenever a `.ref` is created, the reference
   * count increases.
   *
   * You generally don't need to call this function directly since arrays are
   * automatically disposed after being passed into an operation. One common
   * exception is when writing a function and ignoring one of its arguments. In
   * that case, by convention you should dispose of that argument manually.
   *
   * ```
   * function myCustomOperation(a: np.Array, b: np.Array) {
   *   b.dispose(); // Needed to satisfy "move" rules.
   *   return a.add(1);
   * }
   * ```
   */
  abstract dispose(): void;
  /** The shape of the array. */
  get shape(): number[];
  /** The total number of elements in the array. */
  get size(): number;
  /** The dtype of elements stored in the array. */
  get dtype(): DType;
  /**
   * Whether the array is weakly typed.
   *
   * Weakly typed arrays will cast to the dtype of the other operand. See
   * `promoteTypes()` for details.
   */
  get weakType(): boolean;
  /** The number of dimensions of the array. */
  get ndim(): number;
  /** @ignore */
  fullLower(): Tracer;
  neg(): this;
  add(other: this | TracerValue): this;
  mul(other: this | TracerValue): this;
  mod(other: this | TracerValue): this;
  greater(other: this | TracerValue): this;
  less(other: this | TracerValue): this;
  equal(other: this | TracerValue): this;
  notEqual(other: this | TracerValue): this;
  greaterEqual(other: this | TracerValue): this;
  lessEqual(other: this | TracerValue): this;
  /** Sum of the elements of the array over a given axis, or axes. */
  sum(axis?: Axis, opts?: ReduceOpts): this;
  /** Product of the array elements over a given axis. */
  prod(axis?: Axis, opts?: ReduceOpts): this;
  /** Compute the average of the array elements along the specified axis. */
  mean(axis?: Axis, opts?: ReduceOpts): this;
  /** Minimum of the elements of the array along a given axis. */
  min(axis?: Axis, opts?: ReduceOpts): this;
  /** Maximum of the elements of the array along a given axis. */
  max(axis?: Axis, opts?: ReduceOpts): this;
  /** Test whether all array elements along a given axis evaluate to true. */
  all(axis?: Axis, opts?: ReduceOpts): this;
  /** Test whether any array element along a given axis evaluates to true. */
  any(axis?: Axis, opts?: ReduceOpts): this;
  /** Permute the dimensions of an array. Defaults to reversing the axis order. */
  transpose(perm?: number[]): this;
  /**
   * Give a new shape to an array without changing its data.
   *
   * One shape dimension can be -1. In this case, the value is inferred from the
   * length of the array and remaining dimensions.
   */
  reshape(shape: number | number[]): this;
  /** Copy the array and cast to a specified dtype. */
  astype(dtype: DType): this;
  /** Return a bitwise cast of the array, viewed as a new dtype. */
  view(dtype?: DType): this;
  /** Subtract an array from this one. */
  sub(other: this | TracerValue): this;
  /** Divide an array by this one. */
  div(other: this | TracerValue): this;
  /** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
  diagonal(offset?: number, axis1?: number, axis2?: number): this;
  /** Flatten the array without changing its data. */
  flatten(): this;
  /** Flatten the array without changing its data. */
  ravel(): this;
  /**
   * Iterate over the first dimension of this array, returning slices.
   *
   * This can be used to destructure arrays. For example:
   *
   * ```js
   * let x = np.array([[1, 2], [3, 4]]);
   * let [a, b] = x;
   * console.log(a.js()); // [1, 2]
   * console.log(b.js()); // [3, 4]
   * ```
   */
  [Symbol.iterator](): IterableIterator<this>;
  /**
   * Return a sorted copy of an array in ascending order.
   *
   * See `jax.numpy.sort` for full docs.
   */
  sort(axis?: number): this;
  /**
   * Return the indices that would sort an array. Unlike `sort`, this is
   * guaranteed to be a stable sorting algorithm; it always returns the smaller
   * index first in event of ties.
   *
   * See `jax.numpy.argsort` for full docs.
   */
  argsort(axis?: number): this;
  /**
   * Slice an array along one or more axes.
   *
   * This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
   * mimic this in JavaScript, we would write:
   *
   * ```js
   * x.slice([1, 3], 2, [], null);
   * ```
   *
   * The `slice` method accepts a variable number of arguments, each of which
   * can be a number, an empty array, a single-element array, a two-element
   * array, or `null`. The arguments are interpreted as follows:
   *
   * - A number `n` means to access the `n`-th element along that axis, removing
   *   that axis from the resulting shape.
   * - An empty array `[]` means to keep that axis as-is, like `:` in Python.
   * - A single-element array `[i]` means to start slicing from index `i`
   *   (inclusive) to the end of the axis, like `x[i:]`.
   * - A two-element array `[i, j]` means to slice from index `i` (inclusive)
   *   to index `j` (exclusive), like `x[i:j]`.
   * - `null` means to add a new axis at that position, like `np.newaxis`.
   *
   * Like in Python, negative indices are supported, which count from the end of
   * the axis. For example, `-1` means the last element.
   *
   * Strided slices are not yet implemented, so you cannot write `x[::2]` or
   * similar.
   *
   * Advanced indexing by integer arrays is also supported. This translates to
   * the "gather" primitive, and it allows you to access specific elements of
   * the array by integer indices stored in another array.
   */
  slice(...index: (number | [] | [number] | Pair | null | Tracer)[]): this;
}
declare class ShapedArray implements AbstractValue {
  readonly shape: number[];
  readonly dtype: DType;
  readonly weakType: boolean;
  constructor(shape: number[], dtype: DType, weakType: boolean);
  static fromAval(aval: AbstractValue): ShapedArray;
  get ndim(): number;
  get size(): number;
  scalar(): ShapedArray;
  toString(): string;
  equals(other: ShapedArray): boolean;
}
//#endregion
//#region src/frontend/array.d.ts
type ArrayLike = Array | number | boolean;
/**
 * An executable operation that will be dispatched to the backend.
 *
 * This holds a reference to all input buffers used in the operation. After the
 * operation is dispatched, the references should be released.
 */
declare class PendingExecute {
  #private;
  readonly backend: Backend;
  readonly source: Kernel | Routine;
  readonly inputs: Slot[];
  readonly outputs: Slot[];
  prepared: Executable | null;
  submitted: boolean;
  constructor(backend: Backend, source: Kernel | Routine, inputs: Slot[], outputs: Slot[]);
  updateRc(delta: number): void;
  prepare(): Promise<void>;
  prepareSync(): void;
  submit(): void;
}
/** @inline */
type DTypeAndDevice = {
  dtype?: DType;
  device?: Device;
};
/** @inline */
type DTypeShapeAndDevice = {
  dtype?: DType;
  shape?: number[];
  device?: Device;
};
type ArrayConstructorArgs = {
  source: AluExp | Slot;
  st: ShapeTracker;
  dtype: DType;
  weakType: boolean;
  backend: Backend;
  committed: boolean;
  pending?: Iterable<PendingExecute>;
};
/**
 * A multidimensional numeric array with data stored on CPU or GPU.
 *
 * This is the library's core data type. Equivalent to `jax.Array` from JAX, or
 * `torch.Tensor`.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
declare class Array extends Tracer {
  #private;
  /**
   * @ignore
   * Constructs an array from source, shape and backend. Note that if the source
   * is a backend `Slot`, this constructor _takes ownership_ of the slot. It
   * will be freed when the array is disposed.
   */
  constructor(args: ArrayConstructorArgs);
  /** @ignore */
  get aval(): ShapedArray;
  /** Return a simple string representation of the array's dimensions. */
  toString(): string;
  get device(): Device;
  get ref(): this;
  /** Get the current reference count (for debugging memory management). */
  get refCount(): number;
  dispose(): void;
  /**
   * Convert this array into a primitive value.
   *
   * This only works for scalars (0-dimensional arrays). It lets you get values
   * "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
   * evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
   *
   * This method is also called for `==` equality.
   */
  [Symbol.toPrimitive](): any;
  /** Realize the array and return it as data. */
  data(): Promise<DataArray>;
  /**
   * Wait for this array to finish evaluation.
   *
   * Operations and data loading in jax-js are lazy, so this function ensures
   * that pending operations are dispatched and fully executed before it
   * returns.
   *
   * If you are mapping from `data()` or `dataSync()`, it will also trigger
   * dispatch of operations as well.
   *
   * **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
   * asynchronously for multiple arrays.
   */
  blockUntilReady(): Promise<Array>;
  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): DataArray;
  /**
   * Return this array as a WebGPU buffer (with `STORAGE | COPY_SRC`).
   *
   * Only available on the WebGPU backend. The array's memory is still managed
   * by jax-js, and it will be freed when the buffer is no longer in use. You
   * _should not_ mutate the buffer's contents.
   *
   * Note that the GPU buffer may be slightly larger than the array's size; it
   * will always be aligned to 4 bytes.
   */
  gpuBuffer(): Promise<GPUBuffer>;
  /** Synchronous version of `Array.gpuBuffer()`. */
  gpuBufferSync(): GPUBuffer;
  /**
   * Convert this array into a JavaScript object.
   *
   * This is a blocking operation that will compile all of the shaders and wait
   * for execution to complete, synchronously. No other JavaScript code on the
   * site will be run during shader execution.
   *
   * To avoid blocking, prefer `jsAsync()` when possible.
   */
  js(): any;
  /** Convert this array into a JavaScript object, asynchronously. */
  jsAsync(): Promise<any>;
  /**
   * Copy an element of an array to a numeric scalar and return it.
   *
   * Throws an error if the array does not have a single element. The array must
   * either be rank-0, or all dimensions of the shape are 1.
   */
  item(): number;
  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules(): typeof implRules;
  /** @private */
  _realizeSource(): number;
  /** @private Put this array on a new backend, asynchronously. */
  _put(backend: Backend): Promise<Array>;
  /** @private Put this array on a new backend, synchronously. */
  _putSync(backend: Backend): Array;
}
/** Constructor for creating a new array from data. */
declare function array(values: Array | DataArray | RecursiveArray<number> | RecursiveArray<boolean>, {
  shape,
  dtype,
  device
}?: {
  shape?: number[];
} & DTypeAndDevice): Array;
type ImplRule<P extends Primitive> = (tracers: Array[], params: PrimitiveParams<P>) => Array[];
declare const implRules: { [P in Primitive]: ImplRule<P> };
/** Return a new array of given shape and type, filled with zeros. */
declare function zeros(shape: number[], opts?: DTypeAndDevice): Array;
/** Return a new array of given shape and type, filled with ones. */
declare function ones(shape: number[], opts?: DTypeAndDevice): Array;
/** Return a new array of given shape and type, filled with `fill_value`. */
declare function full(shape: number[], fillValue: number | boolean | Array, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/**
 * Create an identity matrix.
 *
 * If numCols is not provided, it defaults to numRows, i.e., a square identity
 * matrix with ones on the diagonal.
 */
declare function eye(numRows: number, numCols?: number, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/** Return the identity matrix, with ones on the main diagonal. */
declare function identity$1(n: number, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/**
 * Return evenly spaced values within a given interval.
 *
 * This can be called with a varying number of arguments, just like the range()
 * builtin function in Python.
 *
 * - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
 * - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
 * - `arange(start, stop, step)` creates an array starting at `start`, ending
 *   before `stop`, with a step size of `step`.
 *
 * Defaults to an integer data type. This can produce unintended results when
 * using a non-integer step, so prefer linspace() in those cases.
 */
declare function arange(start: number, stop?: number, step?: number, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/**
 * Return an array with ones on and below the diagonal and zeros elsewhere.
 *
 * If `k` is provided, it specifies the sub-diagonal on and below which the
 * array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
 * `k>0` is above it.
 */
declare function tri(n: number, m?: number, k?: number, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/** Return the lower triangle of an array. Must be of dimension >= 2. */
declare function tril(a: ArrayLike, k?: number): Array;
/** Return the upper triangle of an array. Must be of dimension >= 2. */
declare function triu(a: ArrayLike, k?: number): Array;
/**
 * Return evenly spaced numbers over a specified interval.
 *
 * Returns _num_ evenly spaced samples, calculated over the interval
 * [`start`, `stop`]. The endpoint `stop` is included in the result by default,
 * but this is controlled by the `endpoint` parameter.
 *
 * The default data type is Float32. Use arange() for integer steps.
 */
declare function linspace(start: number, stop: number, num?: number, endpoint?: boolean, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/**
 * Return numbers spaced evenly on a log scale.
 *
 * In linear space, the sequence starts at `base ** start` and ends at
 * `base ** stop` (see `endpoint` below).
 *
 * @param start - `base ** start` is the starting value of the sequence.
 * @param stop - `base ** stop` is the final value of the sequence, unless `endpoint` is false.
 * @param num - Number of samples to generate. Default is 50.
 * @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
 * @param base - The base of the log space. Default is 10.
 * @returns Array of evenly spaced values on a log scale.
 */
declare function logspace(start: number, stop: number, num?: number, endpoint?: boolean, base?: number, {
  dtype,
  device
}?: DTypeAndDevice): Array;
/**
 * Return numbers spaced evenly on a log scale (a geometric progression).
 *
 * This is similar to logspace(), but with endpoints specified directly. Each
 * output sample is a constant multiple of the previous.
 *
 * Both `start` and `stop` must be nonzero and have the same sign.
 *
 * @param start - The starting value of the sequence.
 * @param stop - The final value of the sequence, unless `endpoint` is false.
 * @param num - Number of samples to generate. Default is 50.
 * @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
 * @returns Array of `num` samples, equally spaced on a log scale.
 */
declare function geomspace(start: number, stop: number, num?: number, endpoint?: boolean, {
  dtype,
  device
}?: DTypeAndDevice): Array;
//#endregion
//#region src/frontend/linearize.d.ts
/** @inline */
type GradOpts = {
  /**
   * Integer or sequence of integers. Specifies which positional argument(s) to
   * differentiate with respect to.
   *
   * Defaults to `0` (the first argument).
   */
  argnums?: number | number[];
  /**
   * The input function returns a pair of `[out, aux]` including an auxiliary
   * value. This `aux` is not differentiated, but is returned alongside the
   * gradient when evaluating the function.
   */
  hasAux?: boolean;
};
declare namespace lax_linalg_d_exports {
  export { cholesky$1 as cholesky, eigh$1 as eigh, lu, svd$1 as svd, triangularSolve };
}
/**
 * Compute the Cholesky decomposition of a symmetric positive-definite matrix.
 *
 * The Cholesky decomposition of a matrix `A` is:
 *
 * - A = L @ L^T  (for upper=false, default)
 * - A = U^T @ U  (for upper=true)
 *
 * where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
 * The input matrix must be symmetric and positive-definite.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const x = np.array([[2., 1.], [1., 2.]]);
 *
 * // Lower Cholesky factorization (default):
 * const L = lax.linalg.cholesky(x);
 * // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
 *
 * // Upper Cholesky factorization:
 * const U = lax.linalg.cholesky(x, { upper: true });
 * // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
 * ```
 */
declare function cholesky$1(a: ArrayLike, {
  upper
}?: {
  upper?: boolean;
}): Array;
/**
 * Eigendecomposition of real symmetric matrices.
 *
 * This uses a cyclic Jacobi routine with an internal convergence check and a
 * fixed maximum number of sweeps.
 *
 * Eigenvectors are returned as columns in the first result, and eigenvalues are
 * returned in ascending order in the second result.
 */
declare function eigh$1(x: ArrayLike, {
  lower,
  sortEigenvalues,
  symmetrizeInput
}?: {
  lower?: boolean;
  sortEigenvalues?: boolean;
  symmetrizeInput?: boolean;
}): [Array, Array];
/**
 * Singular value decomposition of real matrices.
 *
 * This computes a thin SVD using a symmetric eigendecomposition of `A.T @ A`
 * or `A @ A.T`. It is intended as a baseline implementation; it is less
 * numerically stable than a dedicated SVD routine for ill-conditioned inputs.
 *
 * With `computeUv: true`, returns `[u, s, vh]` such that
 * `a ~= u @ diag(s) @ vh`. Only `fullMatrices: false` is supported for
 * non-square matrices.
 */
declare function svd$1(a: ArrayLike, {
  computeUv,
  fullMatrices
}?: {
  computeUv?: boolean;
  fullMatrices?: boolean;
}): [Array, Array, Array] | Array;
/**
 * LU decomposition with partial pivoting.
 *
 * Computes the matrix decomposition: `P @ A = L @ U`, where `P` is a
 * permutation of the rows of `A`, `L` is lower-triangular with unit diagonal,
 * and `U` is upper-triangular.
 *
 * @param x - A batch of matrices with shape `[..., m, n]`.
 *
 * @returns A tuple `(lu, pivots, permutation)` where:
 * - `lu`: combined lower and upper triangular matrices.
 * - `pivots`: an array of pivot indices with shape `[..., min(m, n)]`.
 * - `permutation`: the permutation generated by pivots with shape `[..., m]`.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const A = np.array([[4., 3.], [6., 3.]]);
 * const [lu, pivots, permutation] = lax.linalg.lu(A);
 * // lu ≈ [[6., 3.], [0.6666667, 1.0]]
 * // pivots = [1, 1]
 * // permutation = [1, 0]
 * ```
 */
declare function lu(x: ArrayLike): [Array, Array, Array];
/**
 * Solve a triangular linear system.
 *
 * Solves `a @ x = b` (if leftSide=true) or `x @ a = b` (if leftSide=false)
 * where `a` is a triangular matrix.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const L = np.array([[2., 0.], [1., 3.]]);
 * const b = np.array([4., 7.]).reshape([2, 1]);
 *
 * // Solve L @ x = b
 * const x = lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true });
 * // x = [[2.], [5./3.]]
 * ```
 */
declare function triangularSolve(a: ArrayLike, b: ArrayLike, {
  leftSide,
  lower,
  transposeA,
  unitDiagonal
}?: {
  leftSide?: boolean;
  lower?: boolean;
  transposeA?: boolean;
  unitDiagonal?: boolean;
}): Array;
declare namespace lax_d_exports {
  export { DotDimensionNumbers, PaddingType, bitcastConvertType, conv, convGeneralDilated, convTranspose, convWithGeneralPadding, dot$1 as dot, erf, erfc, lax_linalg_d_exports as linalg, reduceWindow, stopGradient, topK };
}
/** Elementwise bitcast an array into a new dtype. */
declare function bitcastConvertType(x: ArrayLike, newDtype: DType): Array;
/**
 * Dimension numbers for general `dot()` primitive.
 *
 * Contracting dimensions act as a tensor contraction (reduction) along the
 * given axis. They must be the same size in both operands. Batch dimensions
 * are treated as vectorized, leading batch dimensions.
 *
 * The return value has a shape where the first dimensions are shared batch
 * dimensions, followed by `lhs` non-contracting dimensions, followed by
 * `rhs` non-contracting dimensions.
 */
type DotDimensionNumbers = {
  lhsContractingDims?: number[];
  rhsContractingDims?: number[];
  lhsBatchDims?: number[];
  rhsBatchDims?: number[];
};
/**
 * General dot product/contraction operator.
 *
 * Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
 * `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
 */
declare function dot$1(lhs: Array, rhs: Array, {
  lhsContractingDims: lc,
  rhsContractingDims: rc,
  lhsBatchDims: lb,
  rhsBatchDims: rb
}?: DotDimensionNumbers): Array;
type PaddingType = "VALID" | "SAME" | "SAME_LOWER" | Pair[];
/**
 * General n-dimensional convolution operator, with optional dilation.
 *
 * The semantics of this operation mimic the `jax.lax.conv_general_dilated`
 * function in JAX, which wraps XLA's general convolution operator.
 *
 * @param lhs - Input tensor; shape `[N, C_in, ...xs]`
 * @param rhs - Convolution kernel; shape `[C_out, C_in / G, ...ks]`
 * @param windowStrides - Strides for each spatial dimension
 * @param padding - Padding for each spatial dimension, or a string
 *   (`"VALID"`, `"SAME"`, or `"SAME_LOWER"`)
 */
declare function convGeneralDilated(lhs: Array, rhs: Array, windowStrides: number[], padding: PaddingType, {
  lhsDilation,
  rhsDilation,
  featureGroupCount
}?: {
  lhsDilation?: number[];
  rhsDilation?: number[];
  featureGroupCount?: number;
}): Array;
/** Convenience wrapper around `convGeneralDilated`. */
declare function convWithGeneralPadding(lhs: Array, rhs: Array, windowStrides: number[], padding: PaddingType, lhsDilation?: number[], rhsDilation?: number[]): Array;
/** Convenience wrapper around `convGeneralDilated`. */
declare function conv(lhs: Array, rhs: Array, windowStrides: number[], padding: PaddingType): Array;
/**
 * Convenience wrapper for calculating the N-d convolution "transpose".
 *
 * This function directly calculates a fractionally strided conv rather than
 * indirectly calculating the gradient (transpose) of a forward convolution.
 * It is equivalent to the JAX version, except:
 *
 * - The `use_consistent_padding` option is not available. We only have the
 *   consistent padding case (JAX version >0.8.4).
 * - The order of dimensions matches `lax.conv_general_dilated`.
 *
 * Unlike PyTorch/TensorFlow, by default we don't reverse the kernel's spatial
 * dimensions or the `(C_out, C_in)` axis order. To get this behavior, set
 * `transposeKernel` to true.
 *
 * @param lhs - Input tensor; shape `[N, C_in, ...xs]`
 * @param rhs - Convolution kernel; shape `[C_out, C_in, ...ks]`
 * @param strides - Sequence of n integers, sets fractional stride
 * @param padding - Apply padding of `dilation * (kernel_size - 1) - padding` to
 *   each side of the input, so it acts like gradient of `conv()`
 * @param rhsDilation - Atrous dilation for the kernel
 * @param transposeKernel - Flip spatial axes and swap the input/output channels
 *   of the kernel; its shape should be `[C_in, C_out, ...ks]`
 */
declare function convTranspose(lhs: Array, rhs: Array, strides: number[], padding: PaddingType, {
  rhsDilation,
  transposeKernel
}?: {
  rhsDilation?: number[];
  transposeKernel?: boolean;
}): Array;
/** Reduce a computation over padded windows. */
declare function reduceWindow(operand: Array, computation: (x: Array) => Array, windowDimensions: number[], windowStrides?: number[]): Array;
/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
declare function erf(x: ArrayLike): Array;
/**
 * The complementary error function: `erfc(x) = 1 - erf(x)`.
 *
 * This function is more accurate than `1 - erf(x)` for large values of `x`,
 * where `erf(x)` is very close to 1.
 */
declare function erfc(x: ArrayLike): Array;
/**
 * Stops gradient computation.
 *
 * Behaves as the identity function but prevents the flow of gradients during
 * forward or reverse-mode automatic differentiation.
 */
declare function stopGradient(x: ArrayLike): Array;
/**
 * Returns top `k` values and their indices along the specified axis of operand.
 *
 * This is a _stable_ algorithm: If two elements are equal, the lower-index
 * element appears first.
 *
 * @returns A tuple of `(values, indices)`, where `values` and `indices` have
 * the same shape as `x`, except along `axis` where they have size `k`.
 */
declare function topK(x: ArrayLike, k: number, axis?: number): [Array, Array];
declare namespace numpy_fft_d_exports {
  export { ComplexPair, fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn };
}
/**
 * A pair of arrays representing real and imaginary part `a + bj`. Both arrays
 * must have the same shape.
 */
type ComplexPair = {
  real: Array;
  imag: Array;
};
/**
 * Compute a one-dimensional discrete Fourier transform.
 */
declare function fft(a: ComplexPair, axis?: number): ComplexPair;
/**
 * Compute an N-dimensional discrete Fourier transform.
 */
declare function fftn(a: ComplexPair, axes?: number[] | null): ComplexPair;
/**
 * Compute a two-dimensional discrete Fourier transform.
 */
declare function fft2(a: ComplexPair, axes?: number[]): ComplexPair;
/**
 * Compute a one-dimensional inverse discrete Fourier transform.
 */
declare function ifft(a: ComplexPair, axis?: number): ComplexPair;
/**
 * Compute an N-dimensional inverse discrete Fourier transform.
 */
declare function ifftn(a: ComplexPair, axes?: number[] | null): ComplexPair;
/**
 * Compute a two-dimensional inverse discrete Fourier transform.
 */
declare function ifft2(a: ComplexPair, axes?: number[]): ComplexPair;
/**
 * Compute a one-dimensional FFT of real input.
 *
 * The output stores only the non-negative frequency terms.
 */
declare function rfft(a: Array, axis?: number): ComplexPair;
/**
 * Compute the inverse of `rfft`.
 *
 * The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
 * spectrum length.
 */
declare function irfft(a: ComplexPair, axis?: number): Array;
/**
 * Compute an N-dimensional FFT of real input.
 *
 * The final transformed axis stores only the non-negative frequency terms.
 */
declare function rfftn(a: Array, axes?: number[] | null): ComplexPair;
/**
 * Compute a two-dimensional FFT of real input.
 *
 * The final transformed axis stores only the non-negative frequency terms.
 */
declare function rfft2(a: Array, axes?: number[]): ComplexPair;
/**
 * Compute the inverse of `rfftn`.
 *
 * The real output length for the final transformed axis is inferred as
 * `2 * (m - 1)`.
 */
declare function irfftn(a: ComplexPair, axes?: number[] | null): Array;
/**
 * Compute the inverse of `rfft2`.
 */
declare function irfft2(a: ComplexPair, axes?: number[]): Array;
/**
 * Compute the FFT of a Hermitian-symmetric signal.
 *
 * The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
 * Hermitian input length.
 */
declare function hfft(a: ComplexPair, axis?: number): Array;
/** Compute the inverse of `hfft`. */
declare function ihfft(a: Array, axis?: number): ComplexPair;
/** Return sample frequencies for a discrete Fourier transform. */
declare function fftfreq(n: number, d?: number): Array;
/** Return sample frequencies for a real-input discrete Fourier transform. */
declare function rfftfreq(n: number, d?: number): Array;
/** Shift the zero-frequency component to the center of the spectrum. */
declare function fftshift(a: Array, axes?: number | number[] | null): Array;
/** Undo `fftshift`. */
declare function ifftshift(a: Array, axes?: number | number[] | null): Array;
declare namespace numpy_linalg_d_exports {
  export { cholesky, cross$1 as cross, det, diagonal, eigh, eigvalsh, inv, lstsq, matmul, matrixNorm, matrixPower, matrixTranspose, multiDot, norm, outer, slogdet, solve, svd, svdvals, tensordot, trace, vecdot, vectorNorm };
}
/**
 * Compute the Cholesky decomposition of a (batched) positive-definite matrix.
 *
 * This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
 * the input matrix, which is on by default.
 */
declare function cholesky(a: ArrayLike, {
  upper,
  symmetrizeInput
}?: {
  upper?: boolean;
  symmetrizeInput?: boolean;
}): Array;
/**
 * Compute the cross-product of two 3D vectors.
 *
 * This is a simpler and less flexible version of `jax.numpy.cross()`.
 * Both inputs must have size 3 along the specified axis.
 */
declare function cross$1(x1: ArrayLike, x2: ArrayLike, axis?: number): Array;
/** Compute the determinant of a square matrix (batched). */
declare function det(a: ArrayLike): Array;
type EighOpts = {
  symmetrizeInput?: boolean;
};
/**
 * Compute eigenvalues and eigenvectors of real symmetric matrices.
 *
 * The returned eigenvalues are sorted in ascending order, and eigenvectors are
 * returned as columns.
 */
declare function eigh(a: ArrayLike, opts?: EighOpts): [Array, Array];
/** Compute eigenvalues of real symmetric matrices. */
declare function eigvalsh(a: ArrayLike, opts?: EighOpts): Array;
/** @inline */
type SvdOpts = {
  fullMatrices?: boolean;
  computeUv?: boolean;
};
declare function svd(a: ArrayLike, opts: SvdOpts & {
  computeUv: false;
}): Array;
declare function svd(a: ArrayLike, opts?: SvdOpts): [Array, Array, Array];
/** Compute singular values of real matrices. */
declare function svdvals(a: ArrayLike): Array;
/** Compute the inverse of a square matrix (batched). */
declare function inv(a: ArrayLike): Array;
/**
 * Return the least-squares solution to a linear equation.
 *
 * For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
 * For underdetermined systems, this finds the minimum-norm solution for `x`.
 *
 * This currently uses Cholesky decomposition to solve the normal equations,
 * under the hood. The method is not as robust as QR or SVD.
 *
 * @param a coefficient matrix of shape `(M, N)`
 * @param b right-hand side of shape `(M,)` or `(M, K)`
 * @return least-squares solution of shape `(N,)` or `(N, K)`
 */
declare function lstsq(a: ArrayLike, b: ArrayLike): Array;
/** Raise a square matrix to an integer power, via repeated squarings. */
declare function matrixPower(a: ArrayLike, n: number): Array;
/**
 * Compute the dot product of two or more arrays in a single function call,
 * while automatically selecting the fastest evaluation order.
 *
 * If the first argument is 1D, it is treated as a row vector. If the last
 * argument is 1D, it is a column vector. All other intermediate arguments
 * must be 2D matrices.
 *
 * Equivalent to `jax.numpy.einsum()` with the appropriate subscripts.
 */
declare function multiDot(arrays: Array[]): Array;
/** Return sign and natural logarithm of the determinant of `a`. */
declare function slogdet(a: ArrayLike): [Array, Array];
/**
 * Solve a linear system of equations.
 *
 * This solves a (batched) linear system of equations `a @ x = b` for `x` given
 * `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
 *
 * @param a - Coefficient matrix of shape `(..., N, N)`.
 * @param b - Values of shape `(N,)` or `(..., N, M)`.
 * @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
 */
declare function solve(a: ArrayLike, b: ArrayLike): Array;
/**
 * Compute the matrix or vector norm of an array.
 *
 * The behavior depends on the `axis` argument, matching `jax.numpy.linalg.norm()`:
 * - If `axis` is null and `ord` is null, computes the 2-norm of the flattened
 *   array (any rank is allowed).
 * - If `axis` is null and `ord` is given, the input must be 1D (vector norm)
 *   or 2D (matrix norm).
 * - If `axis` is an integer, computes a vector norm over that axis.
 * - If `axis` is a pair of integers, computes a matrix norm over those axes.
 *
 * @param x - Input array.
 * @param ord - Order of the norm (default: 2-norm for vectors, Frobenius norm for matrices).
 * @param axis - Axis or pair of axes to reduce over (default: all axes).
 * @param keepdims - Whether to keep reduced dimensions as size 1.
 * @returns The norm of `x`, reduced over the given axes.
 */
declare function norm(x: ArrayLike, {
  ord,
  axis,
  keepdims
}?: {
  ord?: number | "f" | "fro" | "nuc" | null;
  axis?: number | number[] | null;
  keepdims?: boolean;
}): Array;
/**
 * Compute the vector norm of an array.
 *
 * @param x - Input array.
 * @param ord - Order of the norm (default 2). Supports `Infinity`, `-Infinity`, `0`, or any real number.
 * @param axis - Axis/axes to reduce over (default: all axes).
 * @param keepdims - Whether to keep reduced dimensions as size 1.
 * @returns The norm of `x`, reduced over the given axes.
 */
declare function vectorNorm(x: ArrayLike, {
  ord,
  axis,
  keepdims
}?: {
  ord?: number;
  axis?: number | number[] | null;
  keepdims?: boolean;
}): Array;
/**
 * Calculates the matrix norm of a matrix or a stack of matrices.
 * @param x - Input Array (..., M, N)
 * @param ord - Order of the norm (default "fro").
 * @param keepdims - Whether to keep reduced dimensions as size 1 or collapse axes.
 * @returns Matrix norm of input array. If `keepdims` is `true`, the result has shape (..., 1, 1). Otherwise, it has shape (...,).
 */
declare function matrixNorm(x: ArrayLike, {
  ord,
  keepdims
}?: {
  ord?: number | "fro" | "nuc";
  keepdims?: boolean;
}): Array;
//#endregion
//#region src/library/numpy/dtype-info.d.ts
/** @inline */
type FInfo = Readonly<{
  /** The number of bits occupied by the type. */bits: number; /** Returns the _dtype_ for which finfo returns information. */
  dtype: DType; /** The difference between 1.0 and the next smallest representable float larger than 1.0. */
  eps: number; /** The difference between 1.0 and the next largest representable float smaller than 1.0. */
  epsneg: number; /** The exponent that yields `eps`. */
  machep: number; /** The largest representable finite number. */
  max: number; /** The smallest positive power of the base (2) that causes overflow. */
  maxexp: number; /** The smallest representable (most negative) finite number. */
  min: number; /** The largest negative power of the base (2) without leading zeros in mantissa. */
  minexp: number; /** The exponent that yields `epsneg`. */
  negep: number; /** Number of bits in the exponent portion. */
  nexp: number; /** Number of bits in the mantissa portion. */
  nmant: number; /** The approximate number of decimal digits to which this kind of float is precise. */
  precision: number; /** The approximate decimal resolution, i.e., `10 ** -precision`. */
  resolution: number; /** The smallest positive normal number. */
  smallestNormal: number; /** The smallest positive subnormal number. */
  smallestSubnormal: number;
}>;
/** Machine limits for floating-point types. */
declare function finfo(dtype: DType): FInfo;
/** @inline */
type IInfo = Readonly<{
  /** The number of bits occupied by the type. */bits: number; /** Returns the _dtype_ for which iinfo returns information. */
  dtype: DType; /** The largest representable integer. */
  max: number; /** The smallest representable integer. */
  min: number;
}>;
/** Machine limits for integer types. */
declare function iinfo(dtype: DType): IInfo;
declare namespace numpy_d_exports {
  export { Array, ArrayLike, DType, absolute as abs, absolute, acos, arccosh as acosh, add, all, allclose, any, append, applyAlongAxis, applyOverAxes, arange, acos as arccos, arccosh, asin as arcsin, arcsinh, atan as arctan, atan2 as arctan2, arctanh, argmax, argmin, argsort, round as around, array, arrayEqual, arrayEquiv, arraySplit, asin, arcsinh as asinh, astype, atan, atan2, arctanh as atanh, average, bartlett, bitwiseAnd, bitwiseCount, invert as bitwiseInvert, leftShift as bitwiseLeftShift, invert as bitwiseNot, bitwiseOr, rightShift as bitwiseRightShift, bitwiseXor, blackman, bool, broadcastArrays, broadcastShapes, broadcastTo, cbrt, ceil, clip, columnStack, concatenate, convolve, copysign, corrcoef, correlate, cos, cosh, countNonzero, cov, cross, cumprod, cumsum, cumulativeProd, cumulativeSum, deg2rad, degrees, diag, diagIndices, diagIndicesFrom, diagflat, diagonal, diff, trueDivide as divide, divmod, dot, dsplit, dstack, e, ediff1d, einsum, equal, eulerGamma, exp, exp2, expandDims, expm1, eye, numpy_fft_d_exports as fft, finfo, flip, fliplr, flipud, float16, float32, float64, floatPower, floor, floorDivide, fmax, fmin, fmod, frexp, fromfunction, full, fullLike, geomspace, greater, greaterEqual, hann, heaviside, hsplit, hstack, hypot, identity$1 as identity, iinfo, indices, inf, inner, int32, invert, isfinite, isin, isinf, isnan, isneginf, isposinf, isscalar, ldexp, leftShift, less, lessEqual, numpy_linalg_d_exports as linalg, linspace, log, log10, log1p, log2, logaddexp, logaddexp2, logicalAnd, logicalNot, logicalOr, logicalXor, logspace, matmul, matrixTranspose, matvec, max, maximum, mean, meshgrid, min, minimum, modf, moveaxis, multiply, nan, nanToNum, nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar, ndim, negative, notEqual, ones, onesLike, outer, pad, transpose as permuteDims, pi, polyadd, polyder, polymul, polysub, polyval, positive, power as pow, power, prod, promoteTypes, ptp, rad2deg, radians, ravel, reciprocal, remainder, repeat, reshape, resultType, rightShift, rint, roll, rot90, round, select, shape$1 as shape, sign, signbit, sin, sinc, sinh, size, sort, split$1 as split, sqrt, square, squeeze, stack, std, subtract, sum, swapaxes, take, takeAlongAxis, tan, tanh, tensordot, tile, trace, transpose, trapezoid, tri, tril, triu, trueDivide, trunc, uint32, unstack, unwrap, vander, var_, vdot, vecdot, vecmat, vsplit, vstack, where, zeros, zerosLike };
}
declare const float32 = DType.Float32;
declare const int32 = DType.Int32;
declare const uint32 = DType.Uint32;
declare const bool = DType.Bool;
declare const float16 = DType.Float16;
declare const float64 = DType.Float64;
/** Euler's constant, `e = 2.7182818284590...` */
declare const e: number;
/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
declare const eulerGamma = 0.5772156649015329;
/** Positive infinity. */
declare const inf: number;
/** Floating-point representation of NaN. */
declare const nan: number;
/** This is Pi, `π = 3.14159265358979...` */
declare const pi: number;
/** @function Element-wise addition, with broadcasting. */
declare const add: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Element-wise multiplication, with broadcasting. */
declare const multiply: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Numerical negative of every element of an array. */
declare const negative: (x: ArrayLike) => Array;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
declare const reciprocal: (x: ArrayLike) => Array;
/** @function Round input down to the nearest integer. */
declare const floor: (x: ArrayLike) => Array;
/** @function Round input up to the nearest integer. */
declare const ceil: (x: ArrayLike) => Array;
/** @function Element-wise sine function (takes radians). */
declare const sin: (x: ArrayLike) => Array;
/** @function Element-wise cosine function (takes radians). */
declare const cos: (x: ArrayLike) => Array;
/** @function Element-wise inverse sine function (inverse of sin). */
declare const asin: (x: ArrayLike) => Array;
/** @function Element-wise inverse tangent function (inverse of tan). */
declare const atan: (x: ArrayLike) => Array;
/** @function Calculate the exponential of all elements in the input array. */
declare const exp: (x: ArrayLike) => Array;
/** @function Calculate the natural logarithm of all elements in the input array. */
declare const log: (x: ArrayLike) => Array;
/** @function Calculate the square root of all elements in the input array. */
declare const sqrt: (x: ArrayLike) => Array;
/** @function Return element-wise minimum of the input arrays. */
declare const minimum: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Return element-wise maximum of the input arrays. */
declare const maximum: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const greater: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const less: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const equal: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const notEqual: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const greaterEqual: (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
declare const lessEqual: (x: ArrayLike, y: ArrayLike) => Array;
/** Compute element-wise logical AND. */
declare function logicalAnd(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise logical OR. */
declare function logicalOr(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise logical XOR. */
declare function logicalXor(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise logical NOT. */
declare function logicalNot(x: ArrayLike): Array;
/** Compute element-wise bitwise AND. */
declare function bitwiseAnd(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise bitwise OR. */
declare function bitwiseOr(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise bitwise XOR. */
declare function bitwiseXor(x: ArrayLike, y: ArrayLike): Array;
/**
 * Count the number of 1 bits in the binary representation of the absolute
 * value of each element.
 */
declare function bitwiseCount(x: ArrayLike): Array;
/** Compute element-wise bitwise NOT (inversion). */
declare function invert(x: ArrayLike): Array;
/** Compute element-wise left bit shift. */
declare function leftShift(x: ArrayLike, y: ArrayLike): Array;
/** Compute element-wise right bit shift. */
declare function rightShift(x: ArrayLike, y: ArrayLike): Array;
/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
declare const where: (cond: ArrayLike, x: ArrayLike, y: ArrayLike) => Array;
/**
 * @function
 * Permute the dimensions of an array. Defaults to reversing the axis order.
 */
declare const transpose: (x: ArrayLike, perm?: number[]) => Array;
/**
 * @function
 * Give a new shape to an array without changing its data.
 *
 * One shape dimension can be -1. In this case, the value is inferred from the
 * length of the array and remaining dimensions.
 */
declare const reshape: (x: ArrayLike, shape: number[]) => Array;
/**
 * @function
 * Move axes of an array to new positions. Other axes retain original order.
 */
declare const moveaxis: (x: ArrayLike, src: number, dst: number) => Array;
/**
 * @function
 * Add padding (zeros) to an array.
 *
 * The `width` argument is either an integer or pair of integers, in which case
 * all axes are padded with the same width. Or if it is an array of pairs, each
 * pair specifies the padding for its corresponding axis.
 */
declare const pad: (x: ArrayLike, width: number | Pair | Pair[] | Record<number, Pair>) => Array;
/**
 * @function
 * Return the number of dimensions of an array. Does not consume array reference.
 */
declare const ndim: (x: ArrayLike) => number;
/** @function Return the shape of an array. Does not consume array reference. */
declare const shape$1: (x: ArrayLike) => number[];
/**
 * @function
 * Return an array of zeros with the same shape and type as a given array.
 */
declare const zerosLike: (a: ArrayLike, opts?: DTypeShapeAndDevice) => Array;
/**
 * @function
 * Return an array of ones with the same shape and type as a given array.
 */
declare const onesLike: (a: ArrayLike, opts?: DTypeShapeAndDevice) => Array;
/**
 * @function
 * Return a full array with the same shape and type as a given array.
 */
declare const fullLike: (a: ArrayLike, fillValue: number | boolean | Array, opts?: DTypeShapeAndDevice) => Array;
/**
 * Return the number of elements in an array, optionally along an axis.
 * Does not consume array reference.
 */
declare function size(a: ArrayLike, axis?: number): number;
/**
 * Return true if the input is a scalar.
 *
 * Like JAX (and unlike NumPy), this treats zero-dimensional arrays as
 * scalars. Returns true for numbers, booleans, and 0-dimensional arrays,
 * and false for any other value. Does not consume array reference.
 */
declare function isscalar(element: unknown): boolean;
/** Convert an array to a specified dtype. */
declare function astype(a: ArrayLike, dtype: DType): Array;
/**
 * Return the dtype resulting from applying type promotion rules to the
 * arguments, which may be arrays, JS numbers or booleans, or dtypes.
 *
 * Weakly typed values (e.g., JS numbers) only affect the result when every
 * argument is weakly typed, following the same rules as `promoteTypes()`.
 * Does not consume array references.
 */
declare function resultType(...args: (ArrayLike | DType)[]): DType;
/** Sum of the elements of the array over a given axis, or axes. */
declare function sum(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Sum of the elements of the array over a given axis, or axes, treating NaNs
 * as zero.
 */
declare function nansum(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Count the number of non-zero elements along the specified axis. */
declare function countNonzero(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Product of the array elements over a given axis. */
declare function prod(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Product of the array elements over a given axis, treating NaNs as one. */
declare function nanprod(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Return the minimum of array elements along a given axis. */
declare function min(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Return the maximum of array elements along a given axis. */
declare function max(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Return the maximum of array elements along a given axis, ignoring NaNs.
 *
 * Slices that contain only NaN values return NaN.
 */
declare function nanmax(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Test whether any array element along a given axis evaluates to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
declare function any(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Test whether all array elements along a given axis evaluate to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
declare function all(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Return the peak-to-peak range along a given axis (`max - min`). */
declare function ptp(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Compute the average of the array elements along the specified axis. */
declare function mean(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Compute the average of the array elements along the specified axis, ignoring
 * NaNs. Slices that are all-NaN produce NaN.
 */
declare function nanmean(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * Compute the weighted average along the specified axis.
 *
 * If no axis is specified, mean is computed along all the axes. The weights
 * should have shape matching that of `a`, or if an axis is specified, it should
 * match the shape along those axes.
 */
declare function average(a: ArrayLike, axis?: Axis, opts?: {
  weights?: ArrayLike;
} & ReduceOpts): Array;
/**
 * Returns the indices of the minimum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
declare function argmin(a: ArrayLike, axis?: number, opts?: ReduceOpts): Array;
/**
 * Returns the indices of the minimum values along an axis, ignoring NaNs.
 *
 * By default, index is into the flattened array, otherwise it is along the
 * specified axis. Unlike NumPy, which raises an error for all-NaN slices, this
 * returns -1 for those slices, matching JAX.
 */
declare function nanargmin(a: ArrayLike, axis?: number, opts?: ReduceOpts): Array;
/**
 * Returns the indices of the maximum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
declare function argmax(a: ArrayLike, axis?: number, opts?: ReduceOpts): Array;
/**
 * Returns the indices of the maximum values along an axis, ignoring NaNs.
 *
 * By default, index is into the flattened array, otherwise it is along the
 * specified axis. Unlike NumPy, which raises an error for all-NaN slices, this
 * returns -1 for those slices, matching JAX.
 */
declare function nanargmax(a: ArrayLike, axis?: number, opts?: ReduceOpts): Array;
/** Cumulative sum of elements along an axis. */
declare function cumsum(a: ArrayLike, axis?: number): Array;
/** Alternative API for `jax.numpy.cumsum()`. */
declare function cumulativeSum(x: ArrayLike, opts?: {
  axis?: number;
  includeInitial?: boolean;
}): Array;
/** Cumulative product of elements along an axis. */
declare function cumprod(a: ArrayLike, axis?: number): Array;
/** Alternative API for `jax.numpy.cumprod()`. */
declare function cumulativeProd(x: ArrayLike, opts?: {
  axis?: number;
  includeInitial?: boolean;
}): Array;
/** Cumulative product of elements along an axis, treating NaNs as one. */
declare function nancumprod(a: ArrayLike, axis?: number): Array;
/** Cumulative sum of elements along an axis, treating NaNs as zero. */
declare function nancumsum(a: ArrayLike, axis?: number): Array;
/**
 * Integrate along the given axis using the composite trapezoidal rule.
 *
 * If `x` is provided, the integral is computed with respect to those sample
 * points. Otherwise, the samples are assumed to be evenly spaced with step
 * `dx`. Both `x` and `dx` may be arrays that broadcast against `y`.
 *
 * @param y - Values to integrate.
 * @param x - Optional sample points corresponding to the `y` values.
 * @param opts - Options `dx`, the spacing between samples when `x` is not
 * given (default 1), and `axis`, the axis to integrate along (default -1).
 */
declare function trapezoid(y: ArrayLike, x?: ArrayLike | null, opts?: {
  dx?: ArrayLike;
  axis?: number;
}): Array;
/**
 * Calculate the n-th discrete difference along the given axis.
 *
 * The first difference is given by `out[i] = a[i+1] - a[i]` along the given
 * axis, and higher differences are calculated by applying this recursively.
 * Boolean arrays are differenced with `notEqual` instead of subtraction.
 *
 * The `prepend` and `append` options attach values along `axis` before the
 * differences are taken. Scalars are broadcast to size 1 along `axis`.
 */
declare function diff(a: ArrayLike, n?: number, axis?: number, opts?: {
  prepend?: ArrayLike;
  append?: ArrayLike;
}): Array;
/**
 * Compute the differences between consecutive elements of an array.
 *
 * The input is flattened before taking differences. `opts.toBegin` and
 * `opts.toEnd` are flattened, cast to the input dtype, and prepended or
 * appended to the result.
 */
declare function ediff1d(ary: ArrayLike, opts?: {
  toEnd?: ArrayLike;
  toBegin?: ArrayLike;
}): Array;
/** Reverse the elements in an array along the given axes. */
declare function flip(x: ArrayLike, axis?: Axis): Array;
/**
 * Split an array into multiple sub-arrays along an axis.
 *
 * @param a - The input array to split.
 * @param indicesOrSections - If an integer, it indicates the number of equal
 * sections to create along the specified axis. If a list of integers, it
 * specifies the indices at which to split the array.
 * @param axis - The axis along which to split the array. Default is 0.
 */
declare function split$1(a: ArrayLike, indicesOrSections: number | number[], axis?: number): Array[];
/**
 * Split an array into multiple sub-arrays, allowing unevenly sized sections.
 *
 * If the axis does not divide evenly, the first sections are one element
 * larger than the remaining sections.
 */
declare function arraySplit(a: ArrayLike, indicesOrSections: number | number[], axis?: number): Array[];
/**
 * Split an array into multiple sub-arrays along the depth (third) axis.
 *
 * Equivalent to `split()` with `axis=2`. The input array must have at least
 * three dimensions.
 */
declare function dsplit(a: ArrayLike, indicesOrSections: number | number[]): Array[];
/**
 * Split an array into multiple sub-arrays horizontally (column-wise).
 *
 * Equivalent to `split()` along axis 1 for arrays of rank 2 or higher, or
 * along axis 0 for rank-1 arrays.
 */
declare function hsplit(a: ArrayLike, indicesOrSections: number | number[]): Array[];
/**
 * Split an array into multiple sub-arrays vertically (row-wise).
 *
 * Equivalent to `split` with `axis=0`.
 */
declare function vsplit(a: ArrayLike, indicesOrSections: number | number[]): Array[];
/**
 * Join a sequence of arrays along an existing axis.
 *
 * The arrays must have the same shape, except in the dimension corresponding to
 * `axis` (the first, by default).
 *
 * No scalars can be passed to this function, as the axis is then ambiguous.
 */
declare function concatenate(xs: Array[], axis?: number): Array;
/**
 * Join a sequence of arrays along a new axis.
 *
 * The `axis` parameter specifies the index of the new axis in the dimensions of
 * the result. For example, if `axis=0` it will be the first dimension and if
 * `axis=-1` it will be the last dimension.
 *
 * All shapes must have the same shape.
 */
declare function stack(xs: ArrayLike[], axis?: number): Array;
/**
 * Split an array into a sequence of arrays along the given axis.
 *
 * The `axis` parameter specifies the dimension of the input array to unstack
 * along; it is removed from the shape of each output array. This is the
 * inverse operation of `stack()`.
 */
declare function unstack(x: ArrayLike, axis?: number): Array[];
/**
 * Horizontally stack arrays. Inputs are promoted to rank at least 1, then
 * concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
 */
declare function hstack(xs: ArrayLike[]): Array;
/**
 * Vertically stack arrays. Inputs are promoted to rank at least 2, then
 * concatenated along axis 0.
 */
declare function vstack(xs: ArrayLike[]): Array;
/**
 * Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
 * concatenated along axis 2.
 */
declare function dstack(xs: ArrayLike[]): Array;
/**
 * Stack arrays column-wise. Inputs are promoted to rank at least 2, then
 * concatenated along axis 1.
 */
declare function columnStack(xs: ArrayLike[]): Array;
/** Flip an array vertically (axis=0). */
declare function flipud(x: ArrayLike): Array;
/** Flip an array horizontally (axis=1). */
declare function fliplr(x: ArrayLike): Array;
/** Rotate an array by 90 degrees in the plane specified by axes. */
declare function rot90(m: ArrayLike, k?: number, axes?: number[]): Array;
/**
 * Roll the elements of an array along a specified axis.
 * @param a - Input array.
 * @param shift - The number of places by which elements are shifted. If multiple,
 * the shift is specified for each axis individually.
 * @param axis - The axis or axes to roll. If `null`, the array is flattened,
 * shifted, and then reshaped to its original shape.
 */
declare function roll(a: ArrayLike, shift: number | number[], axis?: Axis): Array;
/** Interchange two axes of an array. */
declare function swapaxes(a: ArrayLike, axis1: number, axis2: number): Array;
/** Transpose the last two dimensions of an array. */
declare function matrixTranspose(a: ArrayLike): Array;
/** Return a 1-D flattened array containing the elements of the input. */
declare function ravel(a: ArrayLike): Array;
/** Remove one or more length-1 axes from an array. */
declare function squeeze(a: ArrayLike, axis?: Axis): Array;
/**
 * Expand the shape of an array by inserting new axes of length 1.
 *
 * @param a - Input array.
 * @param axis - Position(s) in the expanded axes where the new axis (or axes)
 *   is placed. Can be a single integer or an array of integers.
 * @returns Array with the number of dimensions increased.
 *
 * @example
 * ```ts
 * const x = np.array([1, 2]);
 * np.expandDims(x, 0); // Shape [1, 2]
 * np.expandDims(x, 1); // Shape [2, 1]
 * np.expandDims(x, [0, 2]); // Shape [1, 2, 1]
 * ```
 */
declare function expandDims(a: ArrayLike, axis: number | number[]): Array;
/**
 * Repeat each element of an array after themselves.
 *
 * If no axis is provided, use the flattened input array, and return a flat
 * output array.
 */
declare function repeat(a: ArrayLike, repeats: number, axis?: number): Array;
/**
 * Construct an array by repeating A the number of times given by reps.
 *
 * If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
 * integers, the resulting array will have a shape of `(reps[0] * d1,
 * reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
 */
declare function tile(a: ArrayLike, reps: number | number[]): Array;
/**
 * Broadcast an array to a shape, with NumPy-style broadcasing rules.
 *
 * In other words, this lets you append axes to the left, and/or expand
 * dimensions where the shape is 1.
 */
declare function broadcastTo(a: ArrayLike, shape: number[]): Array;
/** Broadcast input shapes to a common output shape. */
declare function broadcastShapes(...shapes: number[][]): number[];
/** Broadcast arrays to a common shape. */
declare function broadcastArrays(...arrays: ArrayLike[]): Array[];
/**
 * Return specified diagonals.
 *
 * If a is 2D, return the diagonal of the array with the given offset. If a is
 * 3D or higher, compute diagonals along the two given axes (default: 0, 1).
 *
 * This returns a view over the existing array. The shape of the resulting array
 * is determined by removing the two axes along which the diagonal is taken,
 * then appending a new axis to the right with holding the diagonals.
 */
declare function diagonal(a: ArrayLike, offset?: number, axis1?: number, axis2?: number): Array;
/**
 * Extract a diagonal or construct a diagonal array.
 *
 * If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
 * array, return a 2D array with v on the k-th diagonal.
 */
declare function diag(v: ArrayLike, k?: number): Array;
/**
 * Create a two-dimensional array with the flattened input on the k-th
 * diagonal.
 */
declare function diagflat(v: ArrayLike, k?: number): Array;
/**
 * Return the indices to access the main diagonal of an array.
 *
 * This returns a list of `ndim` index arrays, each holding `[0, 1, ..., n-1]`.
 * They can be used with advanced indexing to access the main diagonal of an
 * array with `ndim` dimensions, each of length `n`.
 */
declare function diagIndices(n: number, ndim?: number): Array[];
/**
 * Return the indices to access the main diagonal of an array.
 *
 * The input array must be at least 2D, and all dimensions must be of equal
 * length. Returns a list of `ndim` index arrays of dtype `int32`, which can be
 * used to access the main diagonal of the array with `Array.slice()`.
 */
declare function diagIndicesFrom(arr: ArrayLike): Array[];
/** Calculate the sum of the diagonal of an array along the given axes. */
declare function trace(a: ArrayLike, offset?: number, axis1?: number, axis2?: number): Array;
/**
 * Return a sorted copy of an array.
 *
 * The array is sorted along a specified axis (the last by default). This may be
 * an unstable sort, and it dispatches to device-specific implementation.
 */
declare function sort(a: ArrayLike, axis?: number): Array;
/**
 * Return indices that would sort an array. Unlike `sort`, this is guaranteed to
 * be a stable sorting algorithm; it always returns the smaller index first in
 * event of ties.
 *
 * Returns an array of `int32` indices.
 *
 * The array is sorted along a specified axis (the last by default).
 */
declare function argsort(a: ArrayLike, axis?: number): Array;
/**
 * Append values to the end of an array.
 *
 * If `axis` is `null`, both inputs are flattened before concatenation.
 */
declare function append(arr: ArrayLike, values: ArrayLike, axis?: number | null): Array;
/**
 * Take elements from an array along an axis.
 *
 * This is equivalent to advanced indexing with integer indices over that
 * numbered axis. By default, the flattened array is used.
 */
declare function take(a: ArrayLike, indices: ArrayLike, axis?: number | null): Array;
/**
 * Take values from an array by matching 1D index slices along an axis.
 *
 * The input array and indices must have the same rank. Dimensions other than
 * `axis` follow standard broadcasting rules, and the output shape is the
 * broadcasted shape with the size of `indices` along `axis`.
 */
declare function takeAlongAxis(a: ArrayLike, indices: ArrayLike, axis?: number): Array;
/**
 * Select values from a list of choices based on a list of conditions.
 *
 * At each position, the output takes the value of the choice corresponding to
 * the first true condition. Where no condition is true, `defaultValue` is
 * used. Conditions are converted to boolean arrays, and all entries must be
 * mutually broadcast-compatible.
 *
 * @param condlist - List of array-like conditions, converted to booleans.
 * @param choicelist - List of arrays to choose from, same length as
 * `condlist`.
 * @param defaultValue - Value used where every condition is false. Default 0.
 */
declare function select(condlist: ArrayLike[], choicelist: ArrayLike[], defaultValue?: ArrayLike): Array;
/**
 * Apply a function to 1D array slices along an axis.
 *
 * This is implemented internally with applications of `jax.vmap()`, rather than
 * iterating the function over a loop.
 */
declare function applyAlongAxis(func1d: (x: Array) => Array, axis: number, x: ArrayLike): Array;
/** Apply a function repeatedly over specified axes. */
declare function applyOverAxes(func: (x: Array, axis: number) => Array, a: ArrayLike, axes: number[]): Array;
/**
 * Return if two arrays are element-wise equal within a tolerance.
 *
 * The formula used is `|actual - expected| <= atol + rtol * |expected|`, with
 * NaN values comparing equal if `equalNaN` is true.
 */
declare function allclose(actual: Parameters<typeof array>[0], expected: Parameters<typeof array>[0], options?: {
  rtol?: number;
  atol?: number;
  equalNaN?: boolean;
}): boolean;
/**
 * Check if two arrays are element-wise equal.
 *
 * Returns False if the arrays have different shapes. If `equalNaN` is True,
 * NaNs in the same position are considered equal.
 */
declare function arrayEqual(a1: ArrayLike, a2: ArrayLike, opts?: {
  equalNaN?: boolean;
}): Array;
/**
 * Check if two arrays are element-wise equal after broadcasting.
 *
 * Unlike `arrayEqual`, this allows inputs with different but
 * broadcast-compatible shapes.
 */
declare function arrayEquiv(a1: ArrayLike, a2: ArrayLike): Array;
/** Matrix product of two arrays. */
declare function matmul(x: ArrayLike, y: ArrayLike): Array;
/** Matrix-vector product. x1 is [..., M, N], x2 is [..., N] → [..., M]. */
declare function matvec(x1: ArrayLike, x2: ArrayLike): Array;
/** Vector-matrix product. x1 is [..., N], x2 is [..., N, M] → [..., M]. */
declare function vecmat(x1: ArrayLike, x2: ArrayLike): Array;
/** Dot product of two arrays. */
declare function dot(x: ArrayLike, y: ArrayLike): Array;
/**
 * Compute the tensor dot product of two N-dimensional arrays.
 *
 * The behavior is determined by `axes`. If an integer `k`, sum over the last
 * `k` axes of x and the first `k` axes of y. If a tuple, then the first array
 * corresponds to the axes of x and the second to the axes of y.
 */
declare function tensordot(x: ArrayLike, y: ArrayLike, axes?: number | [number[], number[]]): Array;
/**
 * Einstein summation with string subscripts.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@jax-js/jax";
 *
 * const a = np.ones([2, 3]);
 * const b = np.ones([3]);
 * np.einsum("ij,j", a, b); // Shape [2]
 * ```
 */
declare function einsum(subscripts: string, ...operands: ArrayLike[]): Array;
/**
 * Einstein summation alternating between arrays and numeric indices.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@jax-js/jax";
 *
 * const a = np.ones([2, 3]);
 * const b = np.ones([3]);
 * np.einsum(a, [0, 1], b, [1]); // Shape [2]
 * ```
 */
declare function einsum(...args: (ArrayLike | number[])[]): Array;
/**
 * Compute the inner product of two arrays.
 *
 * Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
 * contraction on the last axis.
 *
 * Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
 */
declare function inner(x: ArrayLike, y: ArrayLike): Array;
/**
 * Compute the outer product of two arrays.
 *
 * If the input arrays are not 1D, they will be flattened. Returned array will
 * be of shape `[x.size, y.size]`.
 */
declare function outer(x: ArrayLike, y: ArrayLike): Array;
/**
 * Generate a Vandermonde matrix from a 1D input vector.
 *
 * @param x - Input vector.
 * @param n - Number of columns. Defaults to `x.shape[0]`.
 * @param increasing - If true, powers increase from left to right.
 */
declare function vander(x: ArrayLike, {
  n,
  increasing
}?: {
  n?: number;
  increasing?: boolean;
}): Array;
/**
 * Evaluate a polynomial at specific values.
 *
 * If `p` has length N, this returns the value
 * `p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]`. Additional
 * dimensions in `p` are broadcast against `x`.
 *
 * @param p - Array of polynomial coefficients along the leading axis, from
 *   highest degree to the constant term.
 * @param x - Values at which to evaluate the polynomial.
 */
declare function polyval(p: ArrayLike, x: ArrayLike): Array;
/**
 * Return the sum of two polynomials.
 *
 * The first axis contains polynomial coefficients, ordered from highest degree
 * to the constant term. Remaining axes are broadcast batch dimensions. The
 * shorter coefficient axis is aligned with the end of the longer one.
 */
declare function polyadd(a1: ArrayLike, a2: ArrayLike): Array;
/**
 * Return the difference of two polynomials.
 *
 * The first axis contains polynomial coefficients, ordered from highest degree
 * to the constant term. Remaining axes are broadcast batch dimensions. The
 * shorter coefficient axis is aligned with the end of the longer one.
 */
declare function polysub(a1: ArrayLike, a2: ArrayLike): Array;
/**
 * Return the derivative of the specified order of a polynomial.
 *
 * The first axis contains polynomial coefficients, ordered from highest degree
 * to the constant term. Remaining axes are batch dimensions. Coefficients are
 * promoted to a floating-point dtype. If `m` is at least the number of
 * coefficients, the result is empty, following JAX.
 *
 * @param p - Array of polynomial coefficients along the leading axis.
 * @param m - Order of differentiation, a non-negative integer. Default is 1.
 */
declare function polyder(p: ArrayLike, m?: number): Array;
/**
 * Return the product of two polynomials.
 */
declare function polymul(a1: ArrayLike, a2: ArrayLike): Array;
/**
 * @function Compute the cross product of two arrays.
 *
 * Supports 2D (scalar result) and 3D cross products, with optional axis
 * arguments. If `axis` is given, it overrides `axisa`, `axisb`, and `axisc`.
 */
declare const cross: OwnedFunction<(a: ArrayLike, b: ArrayLike, args_2?: {
  axisa?: number | undefined;
  axisb?: number | undefined;
  axisc?: number | undefined;
  axis?: number | undefined;
} | undefined) => Array>;
/** Vector dot product of two arrays along a given axis. */
declare function vecdot(x: ArrayLike, y: ArrayLike, {
  axis
}?: {
  axis?: number;
}): Array;
/**
 * Return the dot product of two vectors.
 *
 * Like vecdot() but flattens the arguments first into vectors.
 */
declare function vdot(x: ArrayLike, y: ArrayLike): Array;
/** Convolution of two one-dimensional arrays. */
declare function convolve(x: Array, y: Array, mode?: "full" | "same" | "valid"): Array;
/** Correlation of two one dimensional arrays. */
declare function correlate(x: Array, y: Array, mode?: "full" | "same" | "valid"): Array;
/**
 * Return a tuple of coordinate matrices from coordinate vectors.
 *
 * Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
 * fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
 */
declare function meshgrid(xs: Array[], {
  indexing
}?: {
  indexing?: "xy" | "ij";
}): Array[];
/**
 * Return an array representing the indices of a grid.
 *
 * Computes an array where the subarrays contain index values 0, 1, … varying
 * only along the corresponding axis. The result has shape
 * `[dimensions.length, ...dimensions]`.
 *
 * If `sparse` is set, returns one array per dimension instead, each with shape
 * 1 in all dimensions except its own.
 */
declare function indices(dimensions: number[], opts?: DTypeAndDevice & {
  sparse?: false;
}): Array;
declare function indices(dimensions: number[], opts: DTypeAndDevice & {
  sparse: true;
}): Array[];
declare function indices(dimensions: number[], opts: DTypeAndDevice & {
  sparse: boolean;
}): Array | Array[];
/** Construct an array by applying a function over each coordinate. */
declare function fromfunction(func: (...indices: Array[]) => ArrayLike, shape: number[], opts?: DTypeAndDevice): Array;
declare function fromfunction<Tree extends JsTree<ArrayLike>[] | {
  [key: string]: JsTree<ArrayLike>;
}>(func: (...indices: Array[]) => Tree, shape: number[], opts?: DTypeAndDevice): MapJsTree<Tree, ArrayLike, Array>;
/**
 * Clip (limit) the values in an array.
 *
 * Given an interval, values outside the interval are clipped to the interval
 * edges. For example, if an interval of [0, 1] is specified, values smaller
 * than 0 become 0, and values larger than 1 become 1.
 *
 * If either bound is undefined, it is ignored.
 */
declare function clip(a: ArrayLike, min?: ArrayLike, max?: ArrayLike): Array;
/**
 * Calculate the absolute value element-wise.
 *
 * This is the same function as `jax.numpy.abs()`.
 */
declare function absolute(x: ArrayLike): Array;
/**
 * Return an element-wise indication of sign of the input.
 *
 * NaN is returned for NaN input. Every comparison against a NaN is false, so it
 * has to be selected explicitly rather than falling out of the sign tests.
 */
declare function sign(x: ArrayLike): Array;
/**
 * Test element-wise whether the sign bit is set.
 *
 * Unlike comparisons with zero, this distinguishes `-0` from `0`. The sign of
 * NaN is also preserved, except where the platform canonicalizes NaN: JS
 * engines may do so when the cpu backend reads floats out of typed arrays, and
 * conversions of f16 NaN on GPU backends may lose the sign as well.
 */
declare function signbit(x: ArrayLike): Array;
/**
 * @function
 * Return the value with the magnitude of x and the sign of y, element-wise.
 *
 * The sign is read from the sign bit of `y`, so `-0` counts as negative, and a
 * zero or NaN `y` still keeps the magnitude of `x`.
 */
declare const copysign: OwnedFunction<(x: ArrayLike, y: ArrayLike) => Array>;
/**
 * @function
 * Element-wise maximum of two arrays, ignoring NaNs.
 *
 * Unlike `maximum`, if one of the two elements is NaN, the other is returned.
 * NaN is only propagated when both elements are NaN.
 */
declare const fmax: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/**
 * @function
 * Return element-wise minimum of the input arrays, ignoring NaNs.
 *
 * Unlike `minimum()`, if one of the elements is NaN, the other element is
 * returned. NaN is only propagated when both elements are NaN.
 */
declare const fmin: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/** @function Return element-wise positive values of the input (no-op). */
declare const positive: (x: ArrayLike) => Array;
/**
 * Return the Bartlett window of size M, a triangular taper.
 *
 * `w(n) = 1 - |2n - (M-1)| / (M-1)` for `0 <= n <= M-1`.
 */
declare function bartlett(M: number): Array;
/**
 * Return the Hann window of size M, a taper with a weighted cosine bell.
 *
 * `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
 */
declare function hann(M: number): Array;
/**
 * Return the Blackman window of size M, a taper formed with the first three
 * terms of a summation of cosines.
 *
 * `w(n) = 0.42 - 0.5 * cos(2πn/(M-1)) + 0.08 * cos(4πn/(M-1))` for
 * `0 <= n <= M-1`.
 */
declare function blackman(M: number): Array;
/**
 * @function
 * Compute the Heaviside step function. It is defined piecewise:
 * - `heaviside(x1, x2) = 0` for `x1 < 0`,
 * - `heaviside(x1, x2) = x2` for `x1 == 0`,
 * - `heaviside(x1, x2) = 1` for `x1 > 0`.
 */
declare const heaviside: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/** Calculate element-wise square of the input array. */
declare function square(x: ArrayLike): Array;
/** Element-wise tangent function (takes radians). */
declare function tan(x: ArrayLike): Array;
/**
 * @function
 * Return the normalized sinc function.
 *
 * The sinc function is defined as `sin(πx) / (πx)` for `x != 0`, and `1` for `x = 0`.
 * This is the normalized sinc function commonly used in signal processing.
 *
 * **Note:** JVP is not supported at x=0 due to discontinuous derivative. This
 * requires a custom JVP rule to handle properly (see JAX implementation).
 */
declare const sinc: OwnedFunction<(x: ArrayLike) => Array>;
/** Element-wise inverse cosine function (inverse of cos). */
declare function acos(x: ArrayLike): Array;
/**
 * @function
 * Return element-wise hypotenuse for the given legs of a right triangle.
 *
 * In the original NumPy/JAX implementation, this function is more numerically
 * stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
 * stability improvements.
 */
declare const hypot: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/**
 * @function
 * Element-wise arc tangent of y/x with correct quadrant.
 *
 * Returns the angle in radians between the positive x-axis and the point (x, y).
 * The result is in the range [-π, π].
 *
 * Uses numerically stable formulas:
 * - When x >= 0: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
 * - When x < 0:  atan2(y, x) = 2 * atan((sqrt(x^2 + y^2) - x) / y)
 *
 * The output is ill-defined when both x and y are zero.
 */
declare const atan2: OwnedFunction<(y: ArrayLike, x: ArrayLike) => Array>;
/** Element-wise subtraction, with broadcasting. */
declare function subtract(x: ArrayLike, y: ArrayLike): Array;
/** Calculates the floating-point division of x by y element-wise. */
declare function trueDivide(x: ArrayLike, y: ArrayLike): Array;
/**
 * Return the largest integer smaller or equal to the division of the inputs.
 *
 * The result is always rounded towards negative infinity.
 *
 * For floating-point inputs, this is equivalent to `floor(x / y)`.
 * For integer inputs, we use `(x - remainder(x, y)) / y` to handle
 * negative values correctly (note: may overflow near int32 boundaries).
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Element-wise floor division of x by y.
 */
declare function floorDivide(x: ArrayLike, y: ArrayLike): Array;
/**
 * @function
 * Calculate element-wise floating-point modulo operation.
 */
declare const fmod: OwnedFunction<(x: ArrayLike, y: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise remainder of the division (matches sign of y).
 */
declare const remainder: OwnedFunction<(x: ArrayLike, y: ArrayLike) => Array>;
/**
 * Return element-wise quotient and remainder simultaneously.
 *
 * Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Tuple of [quotient, remainder].
 */
declare function divmod(x: ArrayLike, y: ArrayLike): [Array, Array];
/** Round input to the nearest integer towards zero. */
declare function trunc(x: ArrayLike): Array;
/**
 * Return the fractional and integral parts of an array, element-wise.
 *
 * Both parts have the same sign as the input, and satisfy
 * `x == fractional + integral`. Integer inputs are promoted to float32.
 *
 * @param x - Input array.
 * @returns Tuple of [fractional part, integral part].
 */
declare function modf(x: ArrayLike): [Array, Array];
/**
 * @function
 * Round to the given number of decimals.
 *
 * Uses banker's rounding (round half to even) to match NumPy/JAX behavior.
 */
declare const round: OwnedFunction<(a: ArrayLike, decimals?: number | undefined) => Array>;
/**
 * @function
 * Round to the nearest integer, with ties going to the nearest even integer.
 */
declare const rint: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
 *
 * This is the inverse of `frexp()`.
 */
declare function ldexp(x1: ArrayLike, x2: ArrayLike): Array;
/**
 * Decompose floating-point values into mantissa and two's exponent.
 *
 * The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
 * `x != 0`, and the exponent is an integer such that
 * `x = mantissa * 2**exponent`.
 */
declare function frexp(x: ArrayLike): [Array, Array];
/** Calculate `2**p` for all p in the input array. */
declare function exp2(p: ArrayLike): Array;
/** Return the base-2 logarithm of x, element-wise. */
declare function log2(x: ArrayLike): Array;
/** Return the base-10 logarithm of x, element-wise. */
declare function log10(x: ArrayLike): Array;
/** Calculate `exp(x) - 1` element-wise. */
declare function expm1(x: ArrayLike): Array;
/** Calculate the natural logarithm of `1 + x` element-wise. */
declare function log1p(x: ArrayLike): Array;
/**
 * @function
 * Compute `log(exp(x1) + exp(x2))` avoiding overflow.
 */
declare const logaddexp: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/**
 * @function
 * Logarithm of the sum of exponentials of inputs in base-2 avoiding overflow.
 */
declare const logaddexp2: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/** Convert angles from degrees to radians. */
declare function deg2rad(x: ArrayLike): Array;
/** @function Alias of `jax.numpy.deg2rad()`. */
declare const radians: typeof deg2rad;
/** Convert angles from radians to degrees. */
declare function rad2deg(x: ArrayLike): Array;
/** @function Alias of `jax.numpy.rad2deg()`. */
declare const degrees: typeof rad2deg;
/**
 * Unwrap a periodic signal.
 *
 * Shifts values by multiples of `period` to reduce large differences between
 * consecutive values.
 *
 * @param p - Input array.
 * @param discont - Maximum discontinuity between values. Defaults to
 * `period / 2`. Values below `period / 2` are treated as `period / 2`.
 * @param axis - Axis along which unwrap will operate, default is the last axis.
 * @param period - Size of the range over which the input wraps. Default is
 * `2 * pi`.
 */
declare function unwrap(p: ArrayLike, discont?: number | null, axis?: number, period?: number): Array;
/**
 * @function
 * Computes first array raised to power of second array, element-wise.
 */
declare const power: OwnedFunction<(x1: ArrayLike, x2: ArrayLike) => Array>;
/**
 * Computes first array raised to power of second array, element-wise, after
 * promoting inputs to a floating-point dtype.
 */
declare function floatPower(x1: ArrayLike, x2: ArrayLike): Array;
/** @function Calculate the element-wise cube root of the input array. */
declare const cbrt: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise hyperbolic sine of input.
 *
 * `sinh(x) = (exp(x) - exp(-x)) / 2`
 */
declare const sinh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise hyperbolic cosine of input.
 *
 * `cosh(x) = (exp(x) + exp(-x)) / 2`
 */
declare const cosh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise hyperbolic tangent of input.
 *
 * `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 */
declare const tanh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise inverse hyperbolic sine of input.
 *
 * `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
 */
declare const arcsinh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise inverse hyperbolic cosine of input.
 *
 * `arccosh(x) = ln(x + sqrt(x^2 - 1))`
 */
declare const arccosh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Calculate element-wise inverse hyperbolic tangent of input.
 *
 * `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
 */
declare const arctanh: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * Compute the variance of an array.
 *
 * The variance is computed for the flattened array by default, otherwise over
 * the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
declare function var_(x: ArrayLike, axis?: Axis, opts?: {
  mean?: ArrayLike;
  correction?: number;
} & ReduceOpts): Array;
/**
 * Compute the variance of an array along the specified axis, ignoring NaNs.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of non-NaN elements. Slices where
 * `N - correction` is not positive (including all-NaN slices) produce NaN.
 */
declare function nanvar(x: ArrayLike, axis?: Axis, opts?: {
  correction?: number;
} & ReduceOpts): Array;
/**
 * Compute the standard deviation of an array.
 *
 * The standard deviation is computed for the flattened array by default,
 * otherwise over the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
declare function std(x: ArrayLike, axis?: Axis, opts?: {
  mean?: ArrayLike;
  correction?: number;
} & ReduceOpts): Array;
/**
 * Compute the standard deviation of an array along the specified axis,
 * ignoring NaNs.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of non-NaN elements. Slices where
 * `N - correction` is not positive (including all-NaN slices) produce NaN.
 */
declare function nanstd(x: ArrayLike, axis?: Axis, opts?: {
  correction?: number;
} & ReduceOpts): Array;
/** Estimate the sample covariance of a set of variables. */
declare function cov(x: ArrayLike, y?: ArrayLike | null, {
  rowvar
}?: {
  rowvar?: boolean;
}): Array;
/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
declare function corrcoef(x: ArrayLike, y?: ArrayLike): Array;
/**
 * Test whether each element of an array is also present in a second array.
 *
 * Returns a boolean array of the same shape as `element` that is true where
 * the element is in `testElements`, and false otherwise. If `invert` is true,
 * the result is negated, as if by `logicalNot(isin(...))`.
 */
declare function isin(element: ArrayLike, testElements: ArrayLike, opts?: {
  invert?: boolean;
}): Array;
/** Test element-wise for positive or negative infinity, return bool array. */
declare function isinf(x: ArrayLike): Array;
/** Test element-wise for NaN (Not a Number). */
declare function isnan(x: ArrayLike): Array;
/** Test element-wise for negative infinity, return bool array. */
declare function isneginf(x: ArrayLike): Array;
/** Test element-wise for positive infinity, return bool array. */
declare function isposinf(x: ArrayLike): Array;
/**
 * Replace NaN and infinite entries in an array.
 *
 * By default, NaNs are replaced with `0.0`, and infinities are are substituted
 * with the corresponding maximum or minimum finite values.
 */
declare function nanToNum(x: ArrayLike, {
  nan,
  posinf,
  neginf
}?: {
  nan?: ArrayLike;
  posinf?: ArrayLike | null;
  neginf?: ArrayLike | null;
}): Array;
/** Return the minimum of array elements along a given axis, ignoring NaNs. */
declare function nanmin(a: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * @function
 * Test element-wise for finite values (not infinity or NaN).
 */
declare const isfinite: OwnedFunction<(x: ArrayLike) => Array>;
declare namespace nn_d_exports {
  export { celu, dotProductAttention, elu, gelu, glu, hardSigmoid, hardSilu, hardSilu as hardSwish, hardTanh, identity, leakyRelu, log1mexp, logSigmoid, logSoftmax, logmeanexp, logsumexp, mish, oneHot, relu, relu6, selu, sigmoid, silu, softSign, softmax, softplus, sparsePlus, sparseSigmoid, squareplus, standardize, silu as swish };
}
/**
 * Rectified Linear Unit (ReLU) activation function:
 * `relu(x) = max(x, 0)`.
 */
declare function relu(x: ArrayLike): Array;
/**
 * Rectified Linear Unit 6 (ReLU6) activation function:
 * `relu6(x) = min(max(x, 0), 6)`.
 */
declare function relu6(x: ArrayLike): Array;
/**
 * Sigmoid activation function, computed element-wise:
 * `sigmoid(x) = 1 / (1 + exp(-x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Sigmoid_function
 */
declare function sigmoid(x: ArrayLike): Array;
/**
 * Softplus activation function:
 * `softplus(x) = log(1 + exp(x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Softplus
 */
declare function softplus(x: ArrayLike): Array;
/**
 * @function
 * Sparse plus function:
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x+1)**2 / 4`
 * - When `x >= 1`: `x`
 */
declare const sparsePlus: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Sparse sigmoid activation function.
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x + 1) / 2`
 * - When `x >= 1`: `1`
 */
declare const sparseSigmoid: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * Soft-sign activation function, computed element-wise:
 * `softsign(x) = x / (|x| + 1)`.
 */
declare function softSign(x: ArrayLike): Array;
/**
 * @function
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
declare const silu: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * Log-sigmoid activation function, computed element-wise:
 * `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
 */
declare function logSigmoid(x: ArrayLike): Array;
/**
 * @function
 * Identity activation function. Returns the argument unmodified.
 */
declare const identity: (x: ArrayLike) => Array;
/** Leaky rectified linear (ReLU) activation function */
declare function leakyRelu(x: ArrayLike, negativeSlope?: ArrayLike): Array;
/** Hard sigmoid activation function: `relu6(x+3)/6`. */
declare function hardSigmoid(x: ArrayLike): Array;
/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
declare function hardSilu(x: ArrayLike): Array;
/** Hard tanh activation function: `clip(x, -1, 1)`. */
declare function hardTanh(x: ArrayLike): Array;
/**
 * Exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
 */
declare function elu(x: ArrayLike, alpha?: ArrayLike): Array;
/**
 * Continuously-differentiable exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
 */
declare function celu(x: ArrayLike, alpha?: ArrayLike): Array;
/**
 * @function
 * Scaled exponential linear unit activation.
 *
 * Computes the element-wise function:
 * `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
 *
 * Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
 */
declare const selu: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * @function
 * Gaussion error linear unit (GELU) activation function.
 *
 * This is computed element-wise. There are two variants depending on whether
 * `approximate` is set (default true):
 *
 * - Approximate: `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
 * - Exact: `gelu(x) = x * 0.5 * erfc(-x / sqrt(2))`
 *
 * Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
 */
declare const gelu: OwnedFunction<(x: ArrayLike, opts?: {
  approximate?: boolean | undefined;
} | undefined) => Array>;
/**
 * Gated linear unit (GLU) activation function.
 *
 * Splits the `axis` dimension of the input into two halves, a and b, then
 * computes `a * sigmoid(b)`.
 */
declare function glu(x: ArrayLike, axis?: number): Array;
/**
 * Squareplus activation function.
 *
 * Computes the element-wise function:
 * `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
 */
declare function squareplus(x: ArrayLike, b?: ArrayLike): Array;
/**
 * Mish activation function.
 *
 * Computes the element-wise function:
 * `mish(x) = x * tanh(softplus(x))`
 */
declare function mish(x: ArrayLike): Array;
/**
 * Softmax function. Computes the function which rescales elements to the range
 * [0, 1] such that the elements along `axis` sum to 1.
 *
 * If `axis` is not specified, it defaults to the last axis.
 *
 * Reference: https://en.wikipedia.org/wiki/Softmax_function
 */
declare function softmax(x: ArrayLike, axis?: Axis): Array;
/**
 * Log-Softmax function.
 *
 * Computes the logarithm of the `softmax` function, which rescales elements to
 * the range [-infinity, 0).
 *
 * If `axis` is not specified, it defaults to the last axis.
 */
declare function logSoftmax(x: ArrayLike, axis?: Axis): Array;
/**
 * Log-sum-exp reduction. Also a multivariate version of `softplus`.
 *
 * If no axis is specified, the reduction is performed over all elements. This
 * convention differs from `jax.nn.logSoftmax()`.
 *
 * Reference: https://en.wikipedia.org/wiki/LogSumExp
 */
declare function logsumexp(x: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
declare function logmeanexp(x: ArrayLike, axis?: Axis, opts?: ReduceOpts): Array;
/**
 * @function
 * Computes `log(1 - exp(-x))` element-wise, in a numerically stable way.
 *
 * This function is undefined for `x < 0`.
 *
 * Reference: Martin Mächler, "Accurately Computing log(1 - exp(-|a|))
 * Assessed by the Rmpfr package", 2012.
 * https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
 */
declare const log1mexp: OwnedFunction<(x: ArrayLike) => Array>;
/**
 * Standardizes input to zero mean and unit variance.
 *
 * By default, this is computed over the last axis. You can pass in a different
 * axis, or `null` to standardize over all elements.
 *
 * Epsilon is added to denominator, it defaults to `1e-5` for stability.
 */
declare function standardize(x: ArrayLike, axis?: Axis, opts?: {
  mean?: ArrayLike;
  variance?: ArrayLike;
  epsilon?: ArrayLike;
}): Array;
/**
 * One-hot encodes the given indices.
 *
 * Each index in the integer input `x` is encoded as a vector of zeros of length
 * `numClasses`, with a 1 at the index position specified by its value.
 *
 * ```js
 * import { nn, numpy as np } from '@jax-js/jax';
 *
 * nn.oneHot(np.array([1, 1, 2], { dtype: np.int32 }), 3);
 * // Output:
 * // [[0, 1, 0],
 * //  [0, 1, 0],
 * //  [0, 0, 1]]
 * ```
 */
declare function oneHot(x: Array, numClasses: number): Array;
/**
 * Scaled dot product attention (SDPA).
 *
 * Computes `softmax((Q @ K^T) / sqrt(d) + bias) @ V`, where `Q` is the query,
 * `K` is the key, `V` is the value, and `d` is the dimensionality of each key
 * and query vector.
 *
 * Multi-query attention is applied when input `key` and `value` tensors have
 * fewer heads than `query`.
 *
 * We use the following uppercase letters to denote array shapes:
 * - `B` = batch size
 * - `S` = length of key/value sequences (source)
 * - `L` = length of query sequences
 * - `N` = number of attention heads
 * - `H` = dimensionality of each attention head
 * - `K` = number of key/value heads (for grouped-query attention)
 *
 * The batch size `B` may be omitted, which is equivalent to `B = 1`. In this
 * case it must be omitted from all inputs.
 *
 * @param query - Query array; shape `[B, L, N, H]`
 * @param key - Key array; shape `[B, S, K, H]`
 * @param value - Value array; same shape as `key`
 * @param opts.bias - Optional bias to add to the attention logits; shape
 *   `[B, N, L, S]` or broadcastable to it.
 * @param opts.mask - Optional mask to apply to the attention logits; should be
 *   a boolean array broadcastable to `[B, N, L, S]`, where `true` indicates
 *   the element should take part in attention.
 * @param opts.scale - Scaling factor override, default is `1 / sqrt(H)`.
 * @param opts.isCausal - If true, applies a casual mask.
 * @param opts.querySeqLengths - Optional sequence lengths for the queries;
 *   shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.keyValueSeqLengths - Optional sequence lengths for the keys and
 *   values; shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.localWindowSize - If specified, applies a local attention window
 *   of the given size. Can be a single number or a tuple `[left, right]`.
 *
 * @returns The result of the attention operation; shape is the same as query
 *   `[B, L, N, H]`, or `[L, N, H]` if `B` is omitted.
 */
declare function dotProductAttention(query: ArrayLike, key: ArrayLike, value: ArrayLike, opts?: {
  bias?: ArrayLike;
  mask?: ArrayLike;
  scale?: number;
  isCausal?: boolean;
  querySeqLengths?: ArrayLike;
  keyValueSeqLengths?: ArrayLike;
  localWindowSize?: number | [number, number];
}): Array;
declare namespace random_d_exports {
  export { ball, bernoulli, bits, categorical, cauchy, choice, doubleSidedMaxwell, exponential, geometric, gumbel, key, laplace, logistic, lognormal, maxwell, multivariateNormal, normal, pareto, permutation, rademacher, randint, rayleigh, split, triangular, uniform, weibullMin };
}
/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
declare function key(seed: ArrayLike): Array;
/** Splits a PRNG key into `num` new keys by adding a leading axis. */
declare function split(key: Array, num?: number | number[]): Array;
/** Sample uniform bits in the form of unsigned integers. */
declare function bits(key: Array, shape?: number[]): Array;
/**
 * @function
 * Sample uniform random values in [minval, maxval) with given shape.
 */
declare const uniform: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined, args_2?: {
  minval?: number | undefined;
  maxval?: number | undefined;
} | undefined) => Array>;
/**
 * @function
 * Sample points uniformly from the Euclidean unit ball in `d` dimensions.
 *
 * Only the Euclidean `p=2` case is currently supported.
 */
declare const ball: OwnedFunction<(key: ArrayLike, d: number, args_2?: {
  p?: number | undefined;
  shape?: number[] | undefined;
} | undefined) => Array>;
/**
 * Sample Bernoulli random variables with given mean (0,1 categorical).
 *
 * Returns a random Boolean array with the specified shape. `p` can be an array
 * and must be broadcastable to `shape`.
 */
declare function bernoulli(key: Array, p?: ArrayLike, shape?: number[]): Array;
/**
 * @function
 * Sample random values from categorical distributions.
 *
 * Uses the Gumbel max trick for sampling with replacement, or the Gumbel top-k
 * trick for sampling without replacement.
 *
 * Note: Sampling without replacement currently uses argsort and slices the last
 * k elements. This should be replaced with a more efficient topK implementation.
 *
 * - `key` - PRNG key
 * - `logits` - Unnormalized log probabilities of the categorical distribution(s).
 *   `softmax(logits, axis)` gives the corresponding probabilities.
 * - `axis` - Axis along which logits belong to the same categorical distribution.
 * - `shape` - Result batch shape. Must be broadcast-compatible with
 *   `logits.shape` with `axis` removed. Default is `logits.shape` with `axis` removed.
 * - `replace` - If true (default), sample with replacement. If false, sample
 *   without replacement (each category can only be selected once per batch).
 * @returns A random array with int dtype and shape given by `shape` if provided,
 *   otherwise `logits.shape` with `axis` removed.
 */
declare const categorical: OwnedFunction<(key: ArrayLike, logits: ArrayLike, args_2?: {
  axis?: number | undefined;
  shape?: number[] | undefined;
  replace?: boolean | undefined;
} | undefined) => Array>;
/**
 * @function
 * Sample from a Cauchy distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = tan(π * (u - 0.5))` where u ~ Uniform(0, 1).
 */
declare const cauchy: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * Sample from a population with optional replacement and optional probabilities.
 *
 * This implements the common JAX-compatible cases: integer populations and
 * array populations along `axis`. Probabilities `p`, if provided, are sampled
 * via `categorical(log(p))`.
 */
declare function choice(key: Array, a: number | ArrayLike, {
  shape,
  replace,
  p,
  axis
}?: {
  shape?: number[];
  replace?: boolean;
  p?: ArrayLike;
  axis?: number;
}): Array;
/**
 * @function
 * Sample double-sided Maxwell random values with the provided location and scale.
 */
declare const doubleSidedMaxwell: OwnedFunction<(key: ArrayLike, loc: ArrayLike, scale: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample exponential random values according to `p(x) = exp(-x)`.
 */
declare const exponential: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample geometric random values: the number of trials until first success.
 */
declare const geometric: OwnedFunction<(key: ArrayLike, p: ArrayLike, args_2?: {
  shape?: number[] | undefined;
  dtype?: DType | undefined;
} | undefined) => Array>;
/**
 * @function
 * Sample from a Gumbel distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = -log(-log(u))` where u ~ Uniform(0, 1).
 */
declare const gumbel: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample from a Laplace distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: the CDF is `F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))`.
 * Inverting: `x = -sign(u - 0.5) * log(1 - 2 * |u - 0.5|)`.
 */
declare const laplace: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample from a logistic distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = log(u) - log(1-u)`.
 */
declare const logistic: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample log-normal random values: `exp(sigma * normal(key, shape))`.
 */
declare const lognormal: OwnedFunction<(key: ArrayLike, sigma?: ArrayLike | undefined, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample Maxwell-distributed random values.
 */
declare const maxwell: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample multivariate normal random values with given mean and covariance.
 *
 * The values are returned with the given shape, along with the final dimension
 * used to represent the n-dimensional multivariate normal factors.
 *
 * This uses Cholesky decomposition on the covariance matrix.
 *
 * - `key` - PRNG key
 * - `mean` - Mean vector of shape `[..., n]`
 * - `cov` - Covariance of shape `[..., n, n]`, must be positive-definite
 * - `shape` - Result batch shape, must be broadcastable with
 *            `mean.shape[:-1]` and `cov.shape[:-2]`
 * @returns Random samples of shape `[...shape, n]`
 */
declare const multivariateNormal: OwnedFunction<(key: ArrayLike, mean: ArrayLike, cov: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample random values according to `p(x) = 1/sqrt(2pi) * exp(-x^2/2)`.
 *
 * Unlike JAX, this uses the Box-Muller transform. JAX uses the erf_inv primitive instead and
 * directly inverts the CDF, but we don't have support for that yet. Outputs will not be
 * bitwise identical to JAX.
 */
declare const normal: OwnedFunction<(key: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample from a Pareto distribution with shape parameter `b` and support [1, ∞).
 */
declare const pareto: OwnedFunction<(key: ArrayLike, b: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * Return a random permutation of an integer range or of an array along `axis`.
 */
declare function permutation(key: Array, x: number | ArrayLike, axis?: number): Array;
/**
 * @function
 * Sample Rademacher random values, uniformly from {-1, 1}.
 */
declare const rademacher: OwnedFunction<(key: ArrayLike, args_1?: {
  shape?: number[] | undefined;
  dtype?: DType | undefined;
} | undefined) => Array>;
/**
 * @function
 * Sample integer values uniformly from `[minval, maxval)`.
 *
 * This uses modulo reduction of uniform 32-bit random bits. For ranges that do
 * not divide 2^32, this introduces a very small modulo bias.
 */
declare const randint: OwnedFunction<(key: ArrayLike, args_1: {
  minval: number;
  maxval: number;
  shape?: number[] | undefined;
  dtype?: DType | undefined;
}) => Array>;
/**
 * @function
 * Sample Rayleigh random values with the provided scale parameter.
 */
declare const rayleigh: OwnedFunction<(key: ArrayLike, scale?: ArrayLike | undefined, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample triangular random values on `[left, right]` with the given mode.
 */
declare const triangular: OwnedFunction<(key: ArrayLike, left: ArrayLike, mode: ArrayLike, right: ArrayLike, shape?: number[] | undefined) => Array>;
/**
 * @function
 * Sample Weibull minimum random values.
 *
 * Uses `scale * exponential(key) ** (1 / concentration)`.
 */
declare const weibullMin: OwnedFunction<(key: ArrayLike, scale: ArrayLike, concentration: ArrayLike, shape?: number[] | undefined) => Array>;
declare namespace scipy_special_d_exports {
  export { erf, erfc, logSoftmax, logit, logsumexp, softmax };
}
/**
 * @function
 * The logit function, `logit(p) = log(p / (1-p))`.
 */
declare const logit: OwnedFunction<(x: ArrayLike) => Array>;
//#endregion
//#region src/tracing.d.ts
/**
 * Start collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
declare function startTrace(): void;
/**
 * Stop collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
declare function stopTrace(): void;
//#endregion
//#region src/index.d.ts
/** @namespace */
declare const profiler: {
  startTrace: typeof startTrace;
  stopTrace: typeof stopTrace;
};
/**
 * @function
 * Compute the forward-mode Jacobian-vector product for a function.
 */
declare const jvp: <F extends (...args: any[]) => JsTree<Array>, HA extends boolean = false>(f: F, primals: MapJsTree<Parameters<F>, Array, ArrayLike>, tangents: MapJsTree<Parameters<F>, Array, ArrayLike>, opts?: {
  hasAux?: HA;
}) => HA extends true ? ReturnType<F> extends [infer Out, infer Aux] ? [Out, Out, Aux] : never : [ReturnType<F>, ReturnType<F>];
/**
 * @function
 * Vectorize an operation on a batched axis for one or more inputs.
 */
declare const vmap: <F extends (...args: any[]) => JsTree<Array>>(f: F, inAxes?: number | (number | null | JsTree<number | null>)[]) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;
/**
 * @function
 * Compute the Jacobian evaluated column-by-column by forward-mode AD.
 */
declare const jacfwd: <F extends (x: Array) => Array>(f: F) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;
/**
 * @function
 * Construct a Jaxpr by dynamically tracing a function with example inputs.
 */
declare const makeJaxpr: <F extends (...args: any[]) => JsTree<Array>>(f: F) => (...args: Parameters<F>) => {
  jaxpr: ClosedJaxpr;
  treedef: JsTreeDef;
};
/**
 * @function
 * Mark a function for automatic JIT compilation, with operator fusion.
 *
 * The function will be compiled the first time it is called with a set of
 * argument shapes.
 *
 * You can call `.dispose()` on the returned, JIT-compiled function after all
 * calls to free memory associated with array constants.
 *
 * **Options:**
 * - `staticArgnums`: An array of argument indices to treat as static
 *   (compile-time constant). These arguments must be hashable, won't be traced,
 *   and different values will trigger recompilation.
 * - `device`: The device to place the computation on. If not specified, the
 *   computation will be placed on the default device.
 */
declare const jit: <F extends (...args: any[]) => JsTree<Array>>(f: F, opts?: JitOpts) => OwnedFunction<(...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>>;
/**
 * @function
 * Produce a local linear approximation to a function at a point using jvp() and
 * partial evaluation.
 */
declare const linearize: <F extends (...args: any[]) => JsTree<Array>, HA extends boolean = false>(f: F, primals: MapJsTree<Parameters<F>, Array, ArrayLike>, opts?: {
  hasAux?: HA;
}) => HA extends true ? ReturnType<F> extends [infer Out, infer Aux] ? [Out, OwnedFunction<(...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => Out>, Aux] : never : [ReturnType<F>, OwnedFunction<(...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>>];
/**
 * @function
 * Calculate the reverse-mode vector-Jacobian product for a function.
 *
 * The return value is a tuple of `[out, vjpFn]`, where `out` is the output of
 * `f(primals)`, and `vjpFn` is a function that takes in cotangents for each
 * output and returns the cotangents for each input.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return an
 * `[out, aux]` tuple, and `vjp` returns `[out, vjpFn, aux]`.
 *
 * @example
 * ```ts
 * const [y, vjpFn] = vjp(f, [x]);
 *
 * // With hasAux
 * const [y, vjpFn, aux] = vjp(f, [x], { hasAux: true });
 * ```
 */
declare const vjp: <F extends (...args: any[]) => JsTree<Array>, const HA extends boolean = false>(f: F, primals: MapJsTree<Parameters<F>, Array, ArrayLike>, opts?: {
  hasAux?: HA;
}) => HA extends true ? ReturnType<F> extends [infer Out, infer Aux] ? [Out, OwnedFunction<(cotangents: MapJsTree<Out, Array, ArrayLike>) => MapJsTree<Parameters<F>, ArrayLike, Array>>, Aux] : never : [ReturnType<F>, OwnedFunction<(cotangents: MapJsTree<ReturnType<F>, Array, ArrayLike>) => MapJsTree<Parameters<F>, ArrayLike, Array>>];
/** @inline */
type GradOutputType<I, F extends (...args: any[]) => any> = MapJsTree<I extends undefined ? Parameters<F>[0] : I extends number ? Parameters<F>[I] : I extends number[] ? { [K in keyof I]: I[K] extends number ? Parameters<F>[I[K]] : never } : never, ArrayLike, Array>;
/**
 * @function
 * Compute the gradient of a scalar-valued function `f` with respect to its
 * first argument.
 *
 * Pass in different `argnums` to differentiate with respect to other
 * arguments. If a tuple is provided, the return value will be a tuple of
 * gradients corresponding to each argument index.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return a
 * `[out, aux]` tuple, and the return value will be `[gradient, aux]`.
 *
 * @example
 * ```ts
 * const gradient = grad(f)(x);
 *
 * // With `argnums`
 * const [gradientX, gradientZ] = grad(f, { argnums: [0, 2] })(x, y, z);
 *
 * // With `hasAux`
 * const [gradient, aux] = grad(f, { hasAux: true })(x);
 * ```
 */
declare const grad: <F extends (...args: any[]) => JsTree<Array>, const I extends undefined | number | number[] = undefined, const HA extends boolean = false>(f: F, opts?: Omit<GradOpts, "argnums" | "hasAux"> & {
  argnums?: I;
  hasAux?: HA;
}) => (...primals: MapJsTree<Parameters<F>, Array, ArrayLike>) => HA extends true ? ReturnType<F> extends [any, infer Aux] ? [GradOutputType<I, F>, Aux] : never : GradOutputType<I, F>;
/**
 * @function
 * Create a function that evaluates both `f` and the gradient of `f`.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return an
 * `[out, aux]` tuple, and the return value will be `[[out, aux], gradient]`.
 *
 * @example
 * ```ts
 * // Without hasAux
 * const [value, gradient] = valueAndGrad(f)(x);
 *
 * // With hasAux
 * const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);
 * ```
 */
declare const valueAndGrad: <F extends (...args: any[]) => JsTree<Array>, const I extends undefined | number | number[] = undefined, const HA extends boolean = false>(f: F, opts?: Omit<GradOpts, "argnums"> & {
  argnums?: I;
  hasAux?: HA;
}) => (...primals: MapJsTree<Parameters<F>, Array, ArrayLike>) => [ReturnType<F>, GradOutputType<I, F>];
/**
 * @function
 * Compute the Jacobian evaluated row-by-row by reverse-mode AD.
 */
declare const jacrev: typeof jacfwd;
/**
 * @function
 * Compute the Hessian matrix of a scalar-valued function.
 *
 * The Hessian is the matrix of second-order partial derivatives of a function.
 * This is implemented as `jacfwd(grad(f))`.
 *
 * @example
 * ```ts
 * const f = (x: np.Array) => np.sum(x.ref.mul(x.ref).mul(x)); // x^3
 * const H = hessian(f)(np.array([1, 2, 3]));
 * // H[i,j] = d^2f / dx_i dx_j
 * ```
 */
declare const hessian: <F extends (x: Array) => Array>(f: F) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;
/**
 * Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
 *
 * This can be used to wait for the results of an intermediate computation to
 * finish. It's recommended to call this regularly in an iterative computation
 * to avoid queueing up too many pending operations.
 *
 * Does not consume reference to the arrays.
 */
declare function blockUntilReady<T extends JsTree<any>>(x: T): Promise<T>;
/**
 * Transfer `x` to `device`.
 *
 * `x` may be a nested container of arrays or scalars. The resulting structure
 * is committed to the device.
 *
 * If `device` is not specified, this function behaves as identity if the input
 * is already an `Array`, otherwise it places the scalar uncommitted on the
 * default device.
 */
declare function devicePut<T extends JsTree<any>>(x: T, device?: Device): Promise<MapJsTree<T, number | boolean, Array>>;
//#endregion
export { Array, ClosedJaxpr, DType, type Device, Jaxpr, type JsTree, type JsTreeDef, type OwnedFunction, blockUntilReady, defaultDevice, devicePut, devices, getWebGPUDevice, grad, hessian, init, jacfwd, jacrev as jacobian, jacrev, jit, jvp, lax_d_exports as lax, linearize, makeJaxpr, nn_d_exports as nn, numpy_d_exports as numpy, profiler, random_d_exports as random, scipy_special_d_exports as scipySpecial, setDebug, tree_d_exports as tree, valueAndGrad, vjp, vmap };