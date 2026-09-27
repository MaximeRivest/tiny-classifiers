//#region src/pprint.ts
/** General class for pretty-printing expressions with indentation. */
var PPrint = class PPrint {
	indents;
	lines;
	constructor(indents, lines) {
		this.indents = indents;
		this.lines = lines;
	}
	/** Add a fixed amount of indentation to each line. */
	indent(spaces) {
		return new PPrint(this.indents.map((i) => i + spaces), this.lines);
	}
	/** Concatenate pretty-printed expressions with newlines. */
	concat(...items) {
		return new PPrint((this.indents ?? []).concat(...items.map((i) => i.indents)), (this.lines ?? []).concat(...items.map((i) => i.lines)));
	}
	/** Stack one block to the right of another one, sharing 1 common line. */
	stack(other) {
		if (!other.lines.length) return this;
		if (!this.lines.length) return other;
		const indent = this.indents[this.indents.length - 1];
		const s = this.lines[this.lines.length - 1];
		const indentedBlock = other.indent(indent + s.length);
		return new PPrint(this.indents.concat(indentedBlock.indents.slice(1)), this.lines.slice(0, -1).concat(s + " ".repeat(other.indents[0]) + other.lines[0], ...indentedBlock.lines.slice(1)));
	}
	/** Combine this block of lines into a formatted string. */
	toString() {
		return this.lines.map((line, i) => " ".repeat(this.indents[i]) + line).join("\n");
	}
	static pp(s) {
		const lines = s.toString().split("\n");
		return new PPrint(Array(lines.length).fill(0), lines);
	}
};
//#endregion
//#region src/utils.ts
/** @file Generic programming utilities with no dependencies on library code. */
let DEBUG = 0;
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
function setDebug(level) {
	DEBUG = level;
}
function assertNonNull(_value) {}
function unzip2(pairs) {
	const lst1 = [];
	const lst2 = [];
	for (const [x, y] of pairs) {
		lst1.push(x);
		lst2.push(y);
	}
	return [lst1, lst2];
}
function zip(xs, ys) {
	return xs.map((x, i) => [x, ys[i]]);
}
function zipn(...arrays) {
	const minLength = Math.min(...arrays.map((x) => x.length));
	return Array.from({ length: minLength }, (_, i) => arrays.map((arr) => arr[i]));
}
function sorted(arr) {
	return [...arr].sort((a, b) => a - b);
}
function rep(length, value) {
	if (value instanceof Function) return new Array(length).fill(0).map((_, i) => value(i));
	return new Array(length).fill(value);
}
function prod(arr) {
	return arr.reduce((acc, x) => acc * x, 1);
}
function gcd(...values) {
	let a = 0;
	for (let b of values) while (b !== 0) [a, b] = [b, a % b];
	return Math.abs(a);
}
/** Shorthand for integer division, like in Python. */
function intdiv(a, b) {
	return Math.floor(a / b);
}
/** Clamp `x` to the range `[min, max]`. */
function clamp(x, min, max) {
	return Math.max(min, Math.min(max, x));
}
/** Check if two objects are deep equal. */
function deepEqual(a, b) {
	if (a === b) return true;
	if (typeof a !== "object" || typeof b !== "object") return false;
	if (a === null || b === null) return false;
	if (Object.keys(a).length !== Object.keys(b).length) return false;
	for (const key of Object.keys(a)) if (!deepEqual(a[key], b[key])) return false;
	return true;
}
/** Produces a union of maps of sets. This mutates `a`. */
function mapSetUnion(a, b) {
	if (!b) return a;
	for (const [key, setB] of b.entries()) {
		const setA = a.get(key);
		if (setA) for (const val of setB) setA.add(val);
		else a.set(key, setB);
	}
	return a;
}
/** Splits the list based on a condition, `false` first then `true`. */
function partitionList(which, array) {
	const falseList = [];
	const trueList = [];
	for (let i = 0; i < which.length; i++) if (which[i]) trueList.push(array[i]);
	else falseList.push(array[i]);
	return [falseList, trueList];
}
/** Compare two arrays of numbers lexicographically. */
function lexCompare(a, b) {
	const minLength = Math.min(a.length, b.length);
	for (let i = 0; i < minLength; i++) {
		if (a[i] < b[i]) return -1;
		if (a[i] > b[i]) return 1;
	}
	return a.length - b.length;
}
/** Check if an object is a number pair, i.e., a tuple of two numbers. */
function isNumberPair(x) {
	return Array.isArray(x) && x.length === 2 && typeof x[0] === "number" && typeof x[1] === "number";
}
/** Check an axis against number of dimensions, and resolve negative axes. */
function checkAxis(axis, ndim) {
	if (axis < -ndim || axis >= ndim) throw new Error(`Axis ${axis} out of bounds for array of dimension ${ndim}`);
	return axis < 0 ? axis + ndim : axis;
}
/** Normalize common axis argument for functions, defaulting to all axes. */
function normalizeAxis(axis, ndim, sort = true) {
	if (axis === null) return range(ndim);
	else if (typeof axis === "number") return [checkAxis(axis, ndim)];
	else {
		const seen = /* @__PURE__ */ new Set();
		for (const a of axis) {
			const ca = checkAxis(a, ndim);
			if (seen.has(ca)) throw new Error(`Duplicate axis ${ca} passed to function`);
			seen.add(ca);
		}
		return sort ? sorted(seen) : [...seen];
	}
}
/** Check that a shape ends in square matrix dimensions, and return the size. */
function checkSquare(name, shape) {
	if (shape.length < 2 || shape[shape.length - 1] !== shape[shape.length - 2]) throw new Error(`${name}: input must be at least 2D square matrix, got shape ${JSON.stringify(shape)}`);
	return shape[shape.length - 1];
}
/** Check for an array of integers with no duplicates. */
function checkInts(indices) {
	if (typeof indices === "number") {
		if (!Number.isInteger(indices)) throw new TypeError(`Expected integer index, got ${indices}`);
	} else {
		const seen = /* @__PURE__ */ new Set();
		for (const i of indices) {
			if (!Number.isInteger(i)) throw new TypeError(`Expected integer indices, got ${i}`);
			if (seen.has(i)) throw new Error(`Duplicate index ${i} passed to function`);
			seen.add(i);
		}
	}
}
function range(start, stop, step = 1) {
	if (stop === void 0) {
		stop = start;
		start = 0;
	}
	const result = [];
	for (let i = start; i < stop; i += step) result.push(i);
	return result;
}
function isPermutation(axis, n) {
	if (axis.length !== n) return false;
	const seen = /* @__PURE__ */ new Set();
	for (const x of axis) {
		if (x < 0 || x >= n) return false;
		seen.add(x);
	}
	return seen.size === n;
}
function invertPermutation(axis) {
	const n = axis.length;
	if (!isPermutation(axis, n)) throw new Error("invertPermutation: axis is not a permutation");
	const result = new Array(n);
	for (let i = 0; i < n; i++) result[axis[i]] = i;
	return result;
}
/** Topologically sort a DAG, given terminal nodes and an ancestor function. */
function toposort(terminals, parents) {
	const childCounts = /* @__PURE__ */ new Map();
	const stack = [...new Set(terminals)];
	while (true) {
		const node = stack.pop();
		if (!node) break;
		for (const parent of parents(node)) if (childCounts.has(parent)) childCounts.set(parent, childCounts.get(parent) + 1);
		else {
			childCounts.set(parent, 1);
			stack.push(parent);
		}
	}
	for (const node of terminals) childCounts.set(node, childCounts.get(node) - 1);
	const order = [];
	const frontier = terminals.filter((n) => !childCounts.get(n));
	while (true) {
		const node = frontier.pop();
		if (!node) break;
		order.push(node);
		for (const parent of parents(node)) {
			const c = childCounts.get(parent) - 1;
			childCounts.set(parent, c);
			if (c == 0) frontier.push(parent);
		}
	}
	return order.reverse();
}
/**
* Returns the largest power of 2 less than or equal to `max`.
*
* If `hint` is nonzero, it will not return a number greater than the first
* power of 2 that is greater than or equal to `hint`.
*/
function findPow2(hint, max) {
	if (max < 1) throw new Error("max must be a positive integer");
	let ret = 1;
	while ((hint === 0 || ret < hint) && 2 * ret <= max) ret *= 2;
	return ret;
}
/**
* Implements a NumPy-style generalized broadcast rule on two array shapes.
*
* "When operating on two arrays, NumPy compares their shapes element-wise. It
* starts with the trailing (i.e. rightmost) dimension and works its way left.
* Two dimensions are compatible when:
*   1. they are equal, or
*   2. one of them is 1."
*
* Throws a TypeError if the broadcast is not possible.
*
* <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
*/
function generalBroadcast(a, b) {
	const out = [];
	let i = a.length - 1;
	let j = b.length - 1;
	for (; i >= 0 && j >= 0; i--, j--) {
		const x = a[i];
		const y = b[j];
		if (x === y) out.push(x);
		else if (x === 1) out.push(y);
		else if (y === 1) out.push(x);
		else throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
	}
	for (; i >= 0; i--) out.push(a[i]);
	for (; j >= 0; j--) out.push(b[j]);
	return out.reverse();
}
function recursiveFlatten(ar) {
	if (!Array.isArray(ar)) return [ar];
	return ar.flat(Infinity);
}
/** Strip an outermost pair of nested parentheses from an expression, if any. */
function strip1(str) {
	if (str[0] === "(" && str[str.length - 1] === ")") return str.slice(1, -1);
	return str;
}
const _stagingbuf = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
/**
* Polynomial hashes modulo p are good at avoiding collisions in expectation.
* Probability-wise, it's good enough to be used for something like
* deduplicating seen compiler expressions, although it's not adversarial.
*
* See https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(number_theory)
*/
var FpHash = class FpHash {
	value = 8773157n;
	#update(x) {
		const base = 873192869n;
		const modulus = 3189051996290219n;
		this.value = (this.value * base + x) % modulus;
	}
	update(x) {
		if (typeof x === "string") {
			this.#update(BigInt(x.length));
			for (let i = 0; i < x.length; i++) this.#update(BigInt(199 + x.charCodeAt(i)));
		} else if (typeof x === "number") if (Number.isInteger(x)) this.#update(68265653n ^ BigInt(x));
		else {
			_stagingbuf.setFloat64(0, x, true);
			this.#update(_stagingbuf.getBigUint64(0, true));
		}
		else if (typeof x === "boolean") this.#update(x ? 69069841n : 63640693n);
		else if (typeof x === "bigint") this.#update(x ^ 71657401n);
		else if (x === null) this.#update(37832657n);
		else if (x === void 0) this.#update(18145117n);
		else x.hash(this);
		return this;
	}
	static hash(...values) {
		const h = new FpHash();
		for (const x of values) h.update(x);
		return h.value;
	}
};
/** Run a function while caching it inline inside a `Map`. */
function runWithCache(cache, key, thunk) {
	const keyStr = JSON.stringify(key);
	if (cache.has(keyStr)) return cache.get(keyStr);
	else {
		const value = thunk();
		cache.set(keyStr, value);
		return value;
	}
}
/** Async version of `runWithCache`. */
async function runWithCacheAsync(cache, key, thunk) {
	const keyStr = JSON.stringify(key);
	if (cache.has(keyStr)) return cache.get(keyStr);
	else {
		const value = await thunk();
		cache.set(keyStr, value);
		return value;
	}
}
//#endregion
//#region src/alu.ts
/** A numerical data type for array contents. */
let DType = /* @__PURE__ */ function(DType) {
	DType["Float32"] = "float32";
	DType["Int32"] = "int32";
	DType["Uint32"] = "uint32";
	DType["Bool"] = "bool";
	DType["Float16"] = "float16";
	DType["Float64"] = "float64";
	return DType;
}({});
const byteWidth = (dtype) => {
	switch (dtype) {
		case "float32":
		case "int32":
		case "uint32":
		case "bool": return 4;
		case "float16": return 2;
		case "float64": return 8;
		default: throw new TypeError(`Unknown dtype: ${dtype}`);
	}
};
const isFloatDtype = (dtype) => dtype === "float32" || dtype === "float16" || dtype === "float64";
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
function promoteTypes(dtype1, dtype2) {
	if (dtype1 === dtype2) return dtype1;
	const rank = {
		["bool"]: 0,
		["uint32"]: 1,
		["int32"]: 2,
		["float16"]: 3,
		["float32"]: 4,
		["float64"]: 5
	};
	return rank[dtype1] > rank[dtype2] ? dtype1 : dtype2;
}
function dtypedArray(dtype, data) {
	const { buffer, byteLength, byteOffset } = data;
	const length = byteLength / byteWidth(dtype);
	switch (dtype) {
		case "float32": return new Float32Array(buffer, byteOffset, length);
		case "int32":
		case "bool": return new Int32Array(buffer, byteOffset, length);
		case "uint32": return new Uint32Array(buffer, byteOffset, length);
		case "float16": return new Float16Array(buffer, byteOffset, length);
		case "float64": return new Float64Array(buffer, byteOffset, length);
		default: throw new Error(`Unimplemented dtype: ${dtype}`);
	}
}
function dtypedJsArray(dtype, data) {
	switch (dtype) {
		case "float32": return new Float32Array(data);
		case "int32":
		case "bool": return new Int32Array(data);
		case "uint32": return new Uint32Array(data);
		case "float16": return new Float16Array(data);
		case "float64": return new Float64Array(data);
		default: throw new Error(`Unimplemented dtype: ${dtype}`);
	}
}
/**
* Mathematical expression on scalar values.
*
* This is similiar to and based on tinygrad's UOp class, but it's more specific
* to just math on scalars. We're doing this to avoid the complexity of a full
* graph rewrite engine.
*/
var AluExp = class AluExp {
	op;
	dtype;
	src;
	arg;
	#hash;
	#simplified;
	#range;
	constructor(op, dtype, src, arg = void 0) {
		this.op = op;
		this.dtype = dtype;
		this.src = src;
		this.arg = arg;
		if (AluGroup.RequiredFloat.has(op) && !isFloatDtype(dtype)) throw new TypeError(`Unsupported dtype for ${op}: ${dtype}`);
		switch (op) {
			case "Bitcast":
				if (dtype === "bool" || src[0].dtype === "bool" || byteWidth(dtype) !== byteWidth(src[0].dtype)) throw new TypeError(`Bitcast from ${src[0].dtype} -> ${dtype}`);
				break;
			case "Threefry2x32":
				if (dtype !== "uint32" || src.some((x) => x.dtype !== "uint32")) throw new TypeError("Threefry2x32 requires uint32 types");
				break;
			case "BitCombine":
				if (src[0].dtype !== src[1].dtype || isFloatDtype(src[0].dtype)) throw new TypeError(`BitCombine[${arg}] requires matching integral dtype, got ${src[0].dtype} and ${src[1].dtype}`);
				break;
			case "BitShift":
				if (src[0].dtype === "bool" || src[1].dtype === "bool" || isFloatDtype(src[0].dtype) || isFloatDtype(src[1].dtype)) throw new TypeError(`BitShift[${arg}] requires two integral, non-bool dtypes, got ${src[0].dtype} and ${src[1].dtype}`);
				break;
			case "BitInvert":
				if (isFloatDtype(src[0].dtype)) throw new TypeError(`BitInvert requires an integral dtype, got ${src[0].dtype}`);
				break;
			case "BitCount":
				if (dtype !== "int32" || src[0].dtype !== "int32" && src[0].dtype !== "uint32") throw new TypeError(`BitCount requires an int32 or uint32 input and int32 output, got ${src[0].dtype} -> ${dtype}`);
				break;
		}
	}
	static add(a, b) {
		return new AluExp("Add", a.dtype, [a, b]);
	}
	static sub(a, b) {
		return new AluExp("Sub", a.dtype, [a, b]);
	}
	static mul(a, b) {
		return new AluExp("Mul", a.dtype, [a, b]);
	}
	static idiv(a, b) {
		return new AluExp("Idiv", a.dtype, [a, b]);
	}
	static mod(a, b) {
		return new AluExp("Mod", a.dtype, [a, b]);
	}
	static min(a, b) {
		return new AluExp("Min", a.dtype, [a, b]);
	}
	static max(a, b) {
		return new AluExp("Max", a.dtype, [a, b]);
	}
	static bitCombine(a, b, mode) {
		return new AluExp("BitCombine", a.dtype, [a, b], mode);
	}
	static bitShift(a, b, mode) {
		return new AluExp("BitShift", a.dtype, [a, b], mode);
	}
	static bitCount(a) {
		return new AluExp("BitCount", "int32", [a]);
	}
	static sin(a) {
		return new AluExp("Sin", a.dtype, [a]);
	}
	static cos(a) {
		return new AluExp("Cos", a.dtype, [a]);
	}
	static asin(a) {
		return new AluExp("Asin", a.dtype, [a]);
	}
	static atan(a) {
		return new AluExp("Atan", a.dtype, [a]);
	}
	static exp(a) {
		return new AluExp("Exp", a.dtype, [a]);
	}
	static log(a) {
		return new AluExp("Log", a.dtype, [a]);
	}
	static erf(a) {
		return new AluExp("Erf", a.dtype, [a]);
	}
	static erfc(a) {
		return new AluExp("Erfc", a.dtype, [a]);
	}
	static sqrt(a) {
		return new AluExp("Sqrt", a.dtype, [a]);
	}
	static floor(a) {
		if (!isFloatDtype(a.dtype)) return a;
		return new AluExp("Floor", a.dtype, [a]);
	}
	static ceil(a) {
		if (!isFloatDtype(a.dtype)) return a;
		return new AluExp("Ceil", a.dtype, [a]);
	}
	static reciprocal(a) {
		return new AluExp("Reciprocal", a.dtype, [a]);
	}
	static cast(dtype, a) {
		if (a.dtype === dtype) return a;
		return new AluExp("Cast", dtype, [a]);
	}
	static bitcast(dtype, a) {
		if (a.dtype === dtype) return a;
		return new AluExp("Bitcast", dtype, [a]);
	}
	static threefry2x32(k0, k1, c0, c1, mode = "xor") {
		return new AluExp("Threefry2x32", "uint32", [
			k0,
			k1,
			c0,
			c1
		], mode);
	}
	static cmplt(a, b) {
		return new AluExp("Cmplt", "bool", [a, b]);
	}
	static cmpne(a, b) {
		return new AluExp("Cmpne", "bool", [a, b]);
	}
	static where(cond, a, b) {
		return new AluExp("Where", a.dtype, [
			cond,
			a,
			b
		]);
	}
	static const(dtype, value) {
		if (dtype === "bool") value = Number(Boolean(value));
		else if (dtype === "int32") value = Math.trunc(value) | 0;
		else if (dtype === "uint32") value = Math.trunc(value) >>> 0;
		if (typeof value !== "number") throw new TypeError(`Expected a number for constant, got ${typeof value}: ${value}`);
		return new AluExp("Const", dtype, [], value);
	}
	static special(dtype, name, n) {
		return new AluExp("Special", dtype, [], [name, n]);
	}
	static variable(dtype, name) {
		return new AluExp("Variable", dtype, [], name);
	}
	static globalIndex(dtype, gid, len, bufidx) {
		return new AluExp("GlobalIndex", dtype, [bufidx], [gid, len]);
	}
	static globalView(dtype, gid, st, indices) {
		return new AluExp("GlobalView", dtype, indices, [gid, st]);
	}
	static f32(value) {
		return AluExp.const("float32", value);
	}
	static i32(value) {
		return AluExp.const("int32", value);
	}
	static u32(value) {
		return AluExp.const("uint32", value);
	}
	static bool(value) {
		return AluExp.const("bool", Number(value));
	}
	static f16(value) {
		return AluExp.const("float16", value);
	}
	static f64(value) {
		return AluExp.const("float64", value);
	}
	not() {
		if (this.dtype !== "bool") throw new Error("not() can only be called on boolean expressions");
		return AluExp.cmpne(this, AluExp.const("bool", true));
	}
	/** Compute a reasonable expression hash with low collision rate. */
	getHash() {
		if (this.#hash !== void 0) return this.#hash;
		const hasher = new FpHash();
		hasher.update(this.op);
		hasher.update(this.dtype);
		if (this.op === "Const") hasher.update(this.arg);
		else hasher.update(JSON.stringify(this.arg));
		hasher.update(this.src.length);
		for (const s of this.src) hasher.update(s);
		this.#hash = hasher.value;
		return this.#hash;
	}
	hash(state) {
		state.update(this.getHash());
	}
	/** Substitute variables in this AluExp to values. */
	substitute(variables) {
		return this.rewrite((exp) => {
			if (exp.op === "Variable" && Object.hasOwn(variables, exp.arg)) {
				if (exp.dtype !== variables[exp.arg].dtype) throw new Error(`Type mismatch: ${exp.dtype} vs ${variables[exp.arg].dtype}`);
				return variables[exp.arg];
			}
		});
	}
	/** Reindex gid values in this expression as needed. */
	reindexGids(newGids) {
		return this.rewrite((exp) => {
			if (exp.op === "GlobalIndex") {
				const [gid, len] = exp.arg;
				const newGid = newGids[gid];
				if (newGid !== gid) return AluExp.globalIndex(exp.dtype, newGid, len, exp.src[0]);
			} else if (exp.op === "GlobalView") {
				const gid = exp.arg[0];
				const newGid = newGids[gid];
				if (newGid !== gid) return AluExp.globalView(exp.dtype, newGid, exp.arg[1], exp.src);
			}
		});
	}
	#computeRange() {
		if (this.#range !== void 0) return this.#range;
		const src = this.src;
		const minMax4 = (f) => {
			const [r1, r2] = [src[0].#computeRange(), src[1].#computeRange()];
			const values = [
				f(r1[0], r2[0]),
				f(r1[0], r2[1]),
				f(r1[1], r2[0]),
				f(r1[1], r2[1])
			];
			return [Math.min(...values), Math.max(...values)];
		};
		let ret;
		switch (this.op) {
			case "Add":
				ret = [src[0].min + src[1].min, src[0].max + src[1].max];
				break;
			case "Sub":
				ret = [src[0].min - src[1].max, src[0].max - src[1].min];
				break;
			case "Mul":
				ret = minMax4((a, b) => a * b);
				break;
			case "Idiv":
				ret = minMax4((a, b) => Math.trunc(a / b));
				break;
			case "Mod": {
				let divisorRange = src[1].#computeRange();
				if (divisorRange[0] <= 0 && divisorRange[1] >= 0) divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
				if (divisorRange[1] < 0) divisorRange = [-divisorRange[1], -divisorRange[0]];
				const maxDivisor = isFloatDtype(this.dtype) ? divisorRange[1] : divisorRange[1] - 1;
				ret = [clamp(src[0].min, -maxDivisor, 0), clamp(src[0].max, 0, maxDivisor)];
				break;
			}
			case "Min":
				ret = [Math.min(src[0].min, src[1].min), Math.min(src[0].max, src[1].max)];
				break;
			case "Max":
				ret = [Math.max(src[0].min, src[1].min), Math.max(src[0].max, src[1].max)];
				break;
			case "BitCount":
				ret = [0, 32];
				break;
			case "Sin":
				ret = [-1, 1];
				break;
			case "Cos":
				ret = [-1, 1];
				break;
			case "Asin":
				ret = [-Math.PI / 2, Math.PI / 2];
				break;
			case "Atan":
				ret = [-Math.PI / 2, Math.PI / 2];
				break;
			case "Exp":
				ret = [Math.exp(src[0].min), Math.exp(src[0].max)];
				break;
			case "Log":
				ret = [Math.log(src[0].min), Math.log(src[0].max)];
				break;
			case "Erf":
				ret = [erf(src[0].min), erf(src[0].max)];
				break;
			case "Erfc":
				ret = [erfc(src[0].max), erfc(src[0].min)];
				break;
			case "Sqrt":
				ret = [Math.sqrt(src[0].min), Math.sqrt(src[0].max)];
				break;
			case "Floor":
				ret = [Math.floor(src[0].min), Math.floor(src[0].max)];
				break;
			case "Ceil":
				ret = [Math.ceil(src[0].min), Math.ceil(src[0].max)];
				break;
			case "Reciprocal":
				if (src[0].min <= 0 && src[0].max >= 0) return [-Infinity, Infinity];
				ret = [1 / src[0].max, 1 / src[0].min];
				break;
			case "Cast": {
				const wasFloat = isFloatDtype(src[0].dtype);
				const bounded = Number.isFinite(src[0].min) && Number.isFinite(src[0].max);
				if (this.dtype === "bool") {
					const canBeZero = src[0].min <= 0 && src[0].max >= 0;
					ret = src[0].min === 0 && src[0].max === 0 ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
				} else if (this.dtype === "int32") {
					const a = wasFloat ? clamp(src[0].min, -2147483648, 2147483647) | 0 : src[0].min | 0;
					const b = wasFloat ? clamp(src[0].max, -2147483648, 2147483647) | 0 : src[0].max | 0;
					ret = bounded && a <= b ? [a, b] : [-Infinity, Infinity];
				} else if (this.dtype === "uint32") {
					const a = wasFloat ? clamp(src[0].min, 0, 4294967295) >>> 0 : src[0].min >>> 0;
					const b = wasFloat ? clamp(src[0].max, 0, 4294967295) >>> 0 : src[0].max >>> 0;
					ret = bounded && a <= b ? [a, b] : [0, Infinity];
				} else ret = [src[0].min, src[0].max];
				break;
			}
			case "Cmplt":
				ret = [0, 1];
				break;
			case "Cmpne":
				ret = [0, 1];
				break;
			case "Where":
				ret = [Math.min(src[1].min, src[2].min), Math.max(src[1].max, src[2].max)];
				break;
			case "Const":
				ret = [this.arg, this.arg];
				break;
			case "Special":
				ret = [0, this.arg[1] - 1];
				break;
			default: ret = [-Infinity, Infinity];
		}
		if (isNaN(ret[0]) || isNaN(ret[1])) ret = [-Infinity, Infinity];
		if (this.dtype === "bool") {
			ret[0] = clamp(ret[0], 0, 1);
			ret[1] = clamp(ret[1], 0, 1);
		}
		if (this.dtype === "uint32") {
			if (ret[0] < 0 || ret[1] > 4294967295) ret = [0, 4294967295];
		}
		this.#range = ret;
		return ret;
	}
	get min() {
		return this.#computeRange()[0];
	}
	get max() {
		return this.#computeRange()[1];
	}
	/** Largest known integer that divides self. */
	constFactor() {
		if (this.op === "Const") return Math.abs(this.arg);
		if (this.op === "Add") return gcd(this.src[0].constFactor(), this.src[1].constFactor());
		if (this.op === "Mul") {
			if (this.src[0].op === "Const") return Math.abs(this.src[0].arg);
			if (this.src[1].op === "Const") return Math.abs(this.src[1].arg);
		}
		return 1;
	}
	/**
	* Checks if divisible by an integer v and returns the quotient if it is, or
	* `null` if it's not divisible.
	*/
	divides(v) {
		if (v === 1) return this;
		if (this.op === "Const" && this.arg % v === 0) return AluExp.const(this.dtype, this.arg / v);
		if (this.op === "Add") {
			const a = this.src[0].divides(v);
			if (a !== null) {
				const b = this.src[1].divides(v);
				if (b !== null) return AluExp.add(a, b);
			}
		}
		if (this.op === "Mul") {
			const a = this.src[0].divides(v);
			if (a !== null) return AluExp.mul(a, this.src[1]);
			const b = this.src[1].divides(v);
			if (b !== null) return AluExp.mul(this.src[0], b);
		}
		return null;
	}
	#isConstInt() {
		return this.op === "Const" && (this.dtype === "int32" || this.dtype === "uint32");
	}
	/**
	* Get all expressions by deeply matching an operation.
	*
	* For example: `((2+(3*5))+4).splitOp(+) -> [2,(3*5),4]`.
	*/
	*splitOp(sep) {
		if (this.op === sep) for (const src of this.src) yield* src.splitOp(sep);
		else yield this;
	}
	/**
	* Simplify the expression by replacing any known patterns and deduping
	* identical subexpressions.
	*/
	simplify(cache = /* @__PURE__ */ new Map()) {
		if (this.#simplified !== void 0) return this.#simplified;
		const hash = this.getHash();
		const prevCachedValue = cache.get(hash);
		if (prevCachedValue !== void 0) return this.#simplified = prevCachedValue;
		const simplified = this.#simplifyInner(cache);
		const simplifiedHash = simplified.getHash();
		const prevSimplified = cache.get(simplifiedHash);
		if (prevSimplified !== void 0) {
			cache.set(hash, prevSimplified);
			this.#simplified = prevSimplified;
			return prevSimplified;
		} else {
			cache.set(hash, simplified);
			cache.set(simplifiedHash, simplified);
			this.#simplified = simplified;
			return simplified;
		}
	}
	#simplifyInner(cache) {
		const src = this.src.map((x) => x.simplify(cache));
		const { op } = this;
		if (src.every((x) => x.op === "Const") && !AluGroup.Variable.has(op)) {
			const newExp = new AluExp(op, this.dtype, src, this.arg);
			return AluExp.const(this.dtype, newExp.evaluate({}));
		}
		if (op !== "Const" && this.min === this.max) return AluExp.const(this.dtype, this.min);
		if (AluGroup.Binary.has(op)) for (let i = 0; i < 2; i++) {
			if (src[i].op !== "Const") continue;
			const x = src[i].arg;
			if (op === "Add" && x === 0) return src[1 - i];
			if (op === "Sub" && i === 1 && x === 0) return src[1 - i];
			if (op === "Mul" && x === 1) return src[1 - i];
			if (op === "Mul" && x === 0) return AluExp.const(this.dtype, 0);
			if (op === "Idiv" && i === 1 && x === 1 && !isFloatDtype(this.dtype)) return src[1 - i];
			if (op === "Cmpne" && src[i].dtype === "bool" && x === 0) return src[1 - i];
		}
		if ((op === "Add" || op === "Sub") && src[1].op === "Mul") {
			const [a, b] = src[1].src;
			const opNeg = op === "Add" ? "Sub" : "Add";
			if (a.op === "Const" && a.arg === -1) return new AluExp(opNeg, this.dtype, [src[0], b]);
			else if (b.op === "Const" && b.arg === -1) return new AluExp(opNeg, this.dtype, [src[0], a]);
		}
		if (op === "Where" && src.slice(1).every((s, i) => s.op === "Const" && s.arg === 1 - i)) return AluExp.cast(this.dtype, src[0]);
		if (op === "Cmplt") {
			if (src[0].min >= src[1].max) return AluExp.const("bool", false);
			if (src[0].max < src[1].min) return AluExp.const("bool", true);
		}
		if (op === "Cmpne") {
			if (src[0].max < src[1].min || src[0].min > src[1].max) return AluExp.const("bool", true);
		}
		if (op === "Where") {
			if (src[0].max === 0) return src[2];
			if (src[0].min === 1) return src[1];
		}
		if (op === "Mod" && src[1].op === "Const" && src[0].min >= 0 && src[0].max < src[1].arg) return src[0];
		if (op === "Mod" && src[0].op === "Mod" && src[1].#isConstInt() && src[0].src[1].#isConstInt()) {
			const A = src[0].src[1].arg;
			const B = src[1].arg;
			if (A > 0 && B > 0 && (A % B === 0 || B % A === 0)) return AluExp.mod(src[0].src[0], AluExp.const(this.dtype, Math.min(A, B))).simplify();
		}
		if (op === "Add" && src[0].op === "Mul" && src[0].src[1].#isConstInt() && src[1].op === "Mod" && src[1].src[1].#isConstInt() && src[0].src[1].arg === src[1].src[1].arg) {
			const [mul, mod] = src;
			const check = (exp) => {
				return exp.op === "Idiv" && exp.src[1].#isConstInt() && exp.src[1].arg === mod.src[1].arg && exp.src[0] === mod.src[0];
			};
			if (check(mul.src[0])) return mod.src[0];
			if (mul.src[0].op === "Mod") {
				const [x, y] = mul.src[0].src;
				if (check(x)) return AluExp.mod(mod.src[0], AluExp.mul(mod.src[1], y)).simplify(cache);
			}
		}
		if (op === "Idiv" && src[1].#isConstInt()) {
			const [numer, denom] = src;
			const B = denom.arg;
			for (let i = 0; i < 2; i++) {
				if (numer.op === "Mul" && numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					if (A % B === 0) {
						let ret = numer.src[1 - i];
						if (A / B !== 1) ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
						return ret.simplify(cache);
					}
				}
				for (let j = 0; j < 2; j++) if (numer.op === "Add" && numer.src[j].op === "Mul" && numer.src[j].src[i].#isConstInt()) {
					const A = numer.src[j].src[i].arg;
					if (A % B === 0) {
						let ret = numer.src[j].src[1 - i];
						if (A / B !== 1) ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
						const y = numer.src[1 - j];
						const yOverB = y.divides(B);
						if (yOverB !== null) return AluExp.add(ret, yOverB).simplify(cache);
						else if (B > 0 && (ret.min >= 0 && y.min >= 0 || ret.max <= 0 && y.max <= 0)) return AluExp.add(ret, AluExp.idiv(y, AluExp.const(ret.dtype, B))).simplify(cache);
					}
				}
			}
		}
		if (op === "Mod" && src[1].#isConstInt() && src[1].arg > 0 && src[0].min >= 0) {
			const [numer, denom] = src;
			const B = denom.arg;
			for (let i = 0; i < 2; i++) if (numer.op === "Add") {
				if (numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					const x = numer.src[1 - i];
					if (A % B === 0 && x.min >= 0) return AluExp.mod(x, denom).simplify(cache);
				}
				for (let j = 0; j < 2; j++) if (numer.src[i].op === "Mul" && numer.src[i].src[j].#isConstInt()) {
					const A = numer.src[i].src[j].arg;
					const x = numer.src[1 - i];
					if (A % B === 0 && x.min >= 0) return AluExp.mod(x, denom).simplify(cache);
				}
			} else if (numer.op === "Mul") {
				if (numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					if (A % B === 0) return AluExp.const(this.dtype, 0);
					if (A % B === 1) return AluExp.mod(numer.src[1 - i], denom).simplify(cache);
				}
			}
		}
		if ([
			"Add",
			"Mul",
			"Max",
			"Min"
		].includes(op)) {
			const p = (a, b) => new AluExp(op, this.dtype, [a, b]);
			if (src[0].op === "Const") return p(src[1], src[0]).simplify(cache);
			if (src[0].op === op && src[0].src[1].op === "Const") if (src[1].op === "Const") return p(src[0].src[0], p(src[0].src[1], src[1])).simplify(cache);
			else return p(p(src[0].src[0], src[1]), src[0].src[1]).simplify(cache);
			if (src[1].op === op && src[1].src[1].op === "Const") return p(p(src[0], src[1].src[0]), src[1].src[1]).simplify(cache);
		}
		if ((op === "Mod" || op === "Idiv") && src[1].#isConstInt()) {
			const [x, y] = src;
			{
				const factors = [];
				const terms = [];
				for (const u of x.splitOp("Add")) {
					const factor = u.constFactor();
					factors.push(factor);
					terms.push(u.divides(factor));
				}
				const g = gcd(y.arg, ...factors);
				if (g !== 1) {
					let ret = new AluExp(op, this.dtype, [factors.map((f, i) => AluExp.mul(AluExp.const(terms[i].dtype, f / g), terms[i])).reduceRight((a, x) => AluExp.add(x, a)), AluExp.const(y.dtype, y.arg / g)]);
					if (op === "Mod") ret = AluExp.mul(ret, AluExp.const(this.dtype, g));
					return ret.simplify(cache);
				}
			}
			if (y.arg > 0 && x.min >= 0) {
				let [xNoConst, constVal] = [x, 0];
				if (x.op === "Add" && x.src[1].op === "Const") [xNoConst, constVal] = [x.src[0], x.src[1].arg];
				const terms = [];
				const factors = [];
				for (const u of xNoConst.splitOp("Add")) {
					const f = u.constFactor();
					const divided = u.divides(f);
					terms.push(divided ?? u);
					factors.push(divided ? f : 1);
				}
				const quotients = factors.map((f) => Math.floor(f / y.arg));
				const remainders = factors.map((f) => f % y.arg);
				const gcdVal = remainders.reduce((g, r) => gcd(g, r), y.arg);
				if (constVal % y.arg !== constVal || gcdVal !== 1 || remainders.some((r, i) => r === 0 || r !== factors[i] && op === "Mod")) {
					let quo = AluExp.const(x.dtype, Math.floor(constVal / y.arg));
					let rem = AluExp.const(x.dtype, Math.floor(constVal % y.arg / gcdVal));
					for (let i = 0; i < terms.length; i++) if (op === "Idiv" && remainders[i] !== 0) rem = AluExp.add(rem, AluExp.mul(AluExp.const(x.dtype, Math.floor(factors[i] / gcdVal)), terms[i]));
					else {
						rem = AluExp.add(rem, AluExp.mul(AluExp.const(x.dtype, Math.floor(remainders[i] / gcdVal)), terms[i]));
						quo = AluExp.add(quo, AluExp.mul(AluExp.const(x.dtype, quotients[i]), terms[i]));
					}
					if (rem.min >= 0) if (op === "Mod") return AluExp.add(AluExp.mul(AluExp.const(x.dtype, gcdVal), AluExp.mod(rem, AluExp.const(x.dtype, Math.floor(y.arg / gcdVal)))), AluExp.const(x.dtype, constVal % gcdVal)).simplify(cache);
					else return AluExp.add(AluExp.idiv(rem, AluExp.const(x.dtype, Math.floor(y.arg / gcdVal))), quo).simplify(cache);
				}
			}
		}
		return src.every((s, i) => s === this.src[i]) ? this : new AluExp(op, this.dtype, src, this.arg);
	}
	/** Resolve this to a value, or `undefined` if not possible. */
	resolve() {
		const x = this.simplify();
		if (x.op === "Const") return x.arg;
	}
	/**
	* Evaluate the expression on CPU, returning the result.
	*
	* Typically you would compile the AluExp as a representation to a lower-level
	* language. This is just to define the semantics and help debug.
	*
	* Note that the representation of Bool is as a number (0 or 1) here.
	*/
	evaluate(context, globals) {
		if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
			const x = this.src[0].evaluate(context, globals);
			const y = this.src[1].evaluate(context, globals);
			switch (this.op) {
				case "Add": return this.dtype === "bool" ? Number(x || y) : x + y;
				case "Sub": return x - y;
				case "Mul": return this.dtype === "bool" ? Number(x && y) : x * y;
				case "Idiv": return Math.trunc(x / y);
				case "Mod": return x % y;
				case "Min": return Math.min(x, y);
				case "Max": return Math.max(x, y);
				case "BitCombine": {
					let r;
					if (this.arg === "and") r = x & y;
					else if (this.arg === "or") r = x | y;
					else r = x ^ y;
					return this.dtype === "int32" ? r | 0 : r >>> 0;
				}
				case "BitShift":
					if (this.arg === "shl") return this.dtype === "int32" ? x << y | 0 : x << y >>> 0;
					return x >>> y;
				case "Cmplt": return Number(x < y);
				case "Cmpne": return Number(x != y);
				default: throw new Error(`Missing implemementation for ${this.op}`);
			}
		}
		if (AluGroup.Unary.has(this.op)) {
			const x = this.src[0].evaluate(context, globals);
			switch (this.op) {
				case "BitCount": {
					let value = x >>> 0;
					let count = 0;
					while (value !== 0) {
						value = (value & value - 1) >>> 0;
						count++;
					}
					return count;
				}
				case "Sin": return Math.sin(x);
				case "Cos": return Math.cos(x);
				case "Asin": return Math.asin(x);
				case "Atan": return Math.atan(x);
				case "Exp": return Math.exp(x);
				case "Log": return Math.log(x);
				case "Erf": return erf(x);
				case "Erfc": return erfc(x);
				case "Sqrt": return Math.sqrt(x);
				case "Floor": return Math.floor(x);
				case "Ceil": return Math.ceil(x);
				case "Reciprocal": return 1 / x;
				case "Cast": {
					const wasFloat = isFloatDtype(this.src[0].dtype);
					if (this.dtype === "int32") return (wasFloat ? clamp(x, -2147483648, 2147483647) : x) | 0;
					else if (this.dtype === "uint32") return (wasFloat ? clamp(x, 0, 4294967295) : x) >>> 0;
					else if (isFloatDtype(this.dtype)) return x;
					else if (this.dtype === "bool") return Number(Boolean(x));
					else throw new Error(`Unsupported cast to ${this.dtype}`);
				}
				case "Bitcast": {
					const buf = new ArrayBuffer(byteWidth(this.dtype));
					const view = new DataView(buf);
					const fromType = this.src[0].dtype;
					if (fromType === "float32") view.setFloat32(0, x, true);
					else if (fromType === "int32") view.setInt32(0, x, true);
					else if (fromType === "uint32") view.setUint32(0, x, true);
					else if (fromType === "float16") view.setFloat16(0, x, true);
					else if (fromType === "float64") view.setFloat64(0, x, true);
					else throw new Error(`Unsupported bitcast from ${fromType}`);
					if (this.dtype === "float32") return view.getFloat32(0, true);
					else if (this.dtype === "int32") return view.getInt32(0, true);
					else if (this.dtype === "uint32") return view.getUint32(0, true);
					else if (this.dtype === "float16") return view.getFloat16(0, true);
					else if (this.dtype === "float64") return view.getFloat64(0, true);
					else throw new Error(`Unsupported bitcast to ${this.dtype}`);
				}
				default: throw new Error(`Missing implemementation for ${this.op}`);
			}
		}
		switch (this.op) {
			case "Where": return this.src[0].evaluate(context, globals) ? this.src[1].evaluate(context, globals) : this.src[2].evaluate(context, globals);
			case "Threefry2x32": {
				const [k0, k1, c0, c1] = this.src.map((x) => x.evaluate(context, globals));
				const [x0, x1] = threefry2x32(k0, k1, c0, c1);
				if (this.arg === "xor") return (x0 ^ x1) >>> 0;
				else if (this.arg === 0) return x0;
				else if (this.arg === 1) return x1;
				else throw new Error(`Invalid Threefry2x32 mode: ${this.arg}`);
			}
			case "Const": return this.arg;
			case "Special": {
				const x = context[this.arg[0]];
				if (x === void 0) throw new Error(`Missing special: ${this.arg[0]}`);
				return x;
			}
			case "Variable": {
				const x = context[this.arg];
				if (x === void 0) throw new Error(`Missing variable: ${this.arg}`);
				return x;
			}
			case "GlobalIndex": {
				if (!globals) throw new Error("Missing globals function");
				const gid = this.arg[0];
				return globals(gid, this.src[0].evaluate(context, globals));
			}
			case "GlobalView": {
				if (!globals) throw new Error("Missing globals function");
				const gid = this.arg[0];
				const [iexpr, vexpr] = this.arg[1].toAluExp(this.src);
				if (vexpr.evaluate(context, globals)) return globals(gid, iexpr.evaluate(context, globals));
				else return 0;
			}
			default: throw new Error(`Missing implemementation for ${this.op}`);
		}
	}
	/** Get this expression in debug format as a string. */
	toString() {
		const BIN_SYM = {
			["Add"]: "+",
			["Sub"]: "-",
			["Mul"]: "*",
			["Idiv"]: "/",
			["Mod"]: "%"
		};
		const CMP_SYM = {
			["Cmplt"]: "<",
			["Cmpne"]: "!="
		};
		const UNARY_SYM = { ["Reciprocal"]: "1/" };
		return this.fold((node, parts) => {
			switch (node.op) {
				case "Const": return "" + (node.dtype === "bool" ? Boolean(node.arg) : node.arg);
				case "Variable": return `$${node.arg}:${node.dtype}`;
				case "Special": {
					const [name, n] = node.arg;
					return `#${name}{${n}}`;
				}
				case "GlobalIndex": return `G_${node.arg[0]}<${node.dtype}>[${strip1(parts[0])}]`;
				case "GlobalView": {
					const [gid, st] = node.arg;
					const shape = st.shape.join(",");
					const lastStrides = st.lastStrides.join(",");
					const cont = st.contiguous ? "c" : "nc";
					return `GV_${gid}<${node.dtype}>{${shape}:${lastStrides}:${cont}}[${parts.map(strip1).join(", ")}]`;
				}
			}
			if (BIN_SYM[node.op]) return `(${parts[0]} ${BIN_SYM[node.op]} ${parts[1]})`;
			if (CMP_SYM[node.op]) return `(${parts[0]} ${CMP_SYM[node.op]} ${parts[1]})`;
			if (node.op === "BitCombine") {
				const sym = {
					and: "&",
					or: "|",
					xor: "^"
				}[node.arg];
				return `(${parts[0]} ${sym} ${parts[1]})`;
			}
			if (node.op === "BitShift") {
				const sym = node.arg === "shl" ? "<<" : ">>";
				return `(${parts[0]} ${sym} ${parts[1]})`;
			}
			if (UNARY_SYM[node.op]) return `${UNARY_SYM[node.op]}${parts[0]}`;
			if (node.op === "Cast") return `Cast<${node.dtype}>(${strip1(parts[0])})`;
			if (node.op === "Bitcast") return `Bitcast<${node.dtype}>(${strip1(parts[0])})`;
			return `${node.op}(${parts.map(strip1).join(", ")})`;
		});
	}
	/** Generic fold() operation with a reducer over the expression tree. */
	fold(reducer) {
		const visited = /* @__PURE__ */ new Map();
		const recurse = (exp) => {
			if (visited.has(exp)) return visited.get(exp);
			const result = reducer(exp, exp.src.map((s) => recurse(s)));
			visited.set(exp, result);
			return result;
		};
		return recurse(this);
	}
	/** Check if any expression in the tree satisfies a predicate. */
	some(predicate) {
		const visited = /* @__PURE__ */ new Set();
		const recurse = (exp) => {
			if (visited.has(exp)) return false;
			if (predicate(exp)) return true;
			visited.add(exp);
			return exp.src.some(recurse);
		};
		return recurse(this);
	}
	/** Rewrite the expression recursively using a visitor. */
	rewrite(visitor) {
		return this.fold((exp, newSrc) => {
			if (newSrc.length === exp.src.length && newSrc.every((s, i) => s === exp.src[i])) return visitor(exp) ?? exp;
			else {
				const newExp = new AluExp(exp.op, exp.dtype, newSrc, exp.arg);
				return visitor(newExp) ?? newExp;
			}
		});
	}
	/** Collect all nodes that satisfy a predicate. */
	collect(predicate) {
		const result = [];
		this.fold((exp) => {
			if (predicate(exp)) result.push(exp);
		});
		return result;
	}
	/** Produce all distinct AluOp in this expression, with their dtypes. */
	distinctOps() {
		const ops = /* @__PURE__ */ new Map();
		this.fold((exp) => {
			const s = ops.get(exp.op) ?? /* @__PURE__ */ new Set();
			if (!s.has(exp.dtype)) {
				s.add(exp.dtype);
				ops.set(exp.op, s);
			}
		});
		return ops;
	}
	/** Rewrite GlobalView operations to GlobalIndex operations. */
	rewriteGlobalViews() {
		return this.rewrite((exp) => {
			if (exp.op === "GlobalView") {
				const [gid, st] = exp.arg;
				return accessorGlobal(exp.dtype, gid, st, exp.src);
			}
		});
	}
};
const AluGroup = {
	Binary: /* @__PURE__ */ new Set([
		"Add",
		"Sub",
		"Mul",
		"Idiv",
		"Mod",
		"Min",
		"Max",
		"BitCombine",
		"BitShift"
	]),
	Unary: /* @__PURE__ */ new Set([
		"BitCount",
		"Sin",
		"Cos",
		"Asin",
		"Atan",
		"Exp",
		"Log",
		"Erf",
		"Erfc",
		"Sqrt",
		"Floor",
		"Ceil",
		"Reciprocal",
		"Cast",
		"Bitcast"
	]),
	Compare: /* @__PURE__ */ new Set(["Cmplt", "Cmpne"]),
	Variable: /* @__PURE__ */ new Set([
		"Special",
		"Variable",
		"GlobalIndex",
		"GlobalView"
	]),
	Reduce: /* @__PURE__ */ new Set([
		"Add",
		"Mul",
		"Min",
		"Max"
	]),
	RequiredFloat: /* @__PURE__ */ new Set([
		"Sin",
		"Cos",
		"Asin",
		"Atan",
		"Exp",
		"Log",
		"Erf",
		"Erfc",
		"Sqrt",
		"Reciprocal",
		"Floor",
		"Ceil"
	])
};
/** Common variables that can be substituted in expressions. */
const AluVar = {
	gidx: AluExp.variable("int32", "gidx"),
	ridx: AluExp.variable("int32", "ridx"),
	acc: (dtype) => AluExp.variable(dtype, "acc"),
	idx: AluExp.variable("int32", "idx"),
	unroll: AluExp.variable("int32", "unroll"),
	upcast: AluExp.variable("int32", "upcast")
};
/**
* Description of a kernel to be compiled.
*
* Each of these can be processed by a backend into some lower-level
* representation. It consists of one or more fused operations, optionally
* indexing into a buffer.
*/
var Kernel = class {
	nargs;
	size;
	exp;
	reduction;
	constructor(nargs, size, exp, reduction) {
		this.nargs = nargs;
		this.size = size;
		this.exp = exp;
		this.reduction = reduction;
		this.exp = exp.simplify();
	}
	hash(state) {
		state.update(this.nargs).update(this.size).update(this.exp).update(this.reduction);
	}
	pprint() {
		let details = PPrint.pp(`exp = ${this.exp}`);
		details = details.concat(PPrint.pp(`size = ${this.size}`));
		if (this.reduction) details = details.concat(PPrint.pp(`reduction = ${this.reduction}`));
		return PPrint.pp("{ ").stack(details).stack(PPrint.pp(" }"));
	}
	toString() {
		return this.pprint().toString();
	}
	/** The dtype of the values output by this kernel. */
	get dtype() {
		if (this.reduction) return this.reduction.epilogue.dtype;
		else return this.exp.dtype;
	}
	/** The number of bytes in the output array when evaluating this kernel. */
	get bytes() {
		return this.size * byteWidth(this.dtype);
	}
};
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
var Reduction = class {
	dtype;
	op;
	size;
	epilogue;
	constructor(dtype, op, size, epilogue = AluVar.acc(dtype)) {
		this.dtype = dtype;
		this.op = op;
		this.size = size;
		this.epilogue = epilogue;
		if (!AluGroup.Reduce.has(op)) throw new TypeError(`Unsupported reduction: ${op}`);
		this.epilogue = epilogue.simplify();
		if (this.dtype === "float16" && this.op === "Add") {
			this.epilogue = this.epilogue.substitute({ acc: AluExp.cast(this.dtype, AluVar.acc("float32")) });
			this.dtype = "float32";
		}
	}
	hash(state) {
		state.update(this.dtype).update(this.op).update(this.size).update(this.epilogue);
	}
	toString() {
		return `${this.op}{${this.size}} -> ${this.epilogue}`;
	}
	/** Get the identity for this reduction operation. */
	get identity() {
		if (this.dtype === "bool") return this.op === "Add" || this.op === "Max" ? 0 : 1;
		else if (this.dtype === "int32") {
			if (this.op === "Add") return 0;
			else if (this.op === "Mul") return 1;
			else if (this.op === "Min") return -1 >>> 1;
			else if (this.op === "Max") return 1 << 31;
		} else if (this.dtype === "uint32") {
			if (this.op === "Add") return 0;
			else if (this.op === "Mul") return 1;
			else if (this.op === "Min") return -1 >>> 0;
			else if (this.op === "Max") return 0;
		} else if (isFloatDtype(this.dtype)) {
			if (this.op === "Add") return 0;
			else if (this.op === "Mul") return 1;
			else if (this.op === "Min") return Infinity;
			else if (this.op === "Max") return -Infinity;
		}
		throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
	}
	/** Evaluate this operation on CPU. */
	evaluate(...values) {
		if (this.dtype === "bool") {
			if (this.op === "Add" || this.op === "Max") return values.reduce((a, b) => a || b, false);
			else if (this.op === "Mul" || this.op === "Min") return values.reduce((a, b) => a && b, true);
		} else if (this.dtype === "int32") {
			if (this.op === "Add") return values.reduce((a, b) => a + b | 0, 0);
			else if (this.op === "Mul") return values.reduce((a, b) => a * b | 0, 1);
			else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), -1 >>> 1);
			else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), 1 << 31);
		} else if (this.dtype === "uint32") {
			if (this.op === "Add") return values.reduce((a, b) => a + b >>> 0, 0);
			else if (this.op === "Mul") return values.reduce((a, b) => a * b >>> 0, 1);
			else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), -1 >>> 0);
			else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), 0);
		} else if (isFloatDtype(this.dtype)) {
			if (this.op === "Add") return values.reduce((a, b) => a + b, 0);
			else if (this.op === "Mul") return values.reduce((a, b) => a * b, 1);
			else if (this.op === "Min") return values.reduce((a, b) => Math.min(a, b), Infinity);
			else if (this.op === "Max") return values.reduce((a, b) => Math.max(a, b), -Infinity);
		}
		throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
	}
};
/** Expression for accessing `indices` in input array with the given shape. */
function accessorGlobal(dtype, gid, st, indices) {
	const [index, valid] = st.toAluExp(indices);
	const [, len] = st.views[0].dataRange();
	if (valid.resolve()) return AluExp.globalIndex(dtype, gid, len, index);
	return AluExp.where(valid, AluExp.globalIndex(dtype, gid, len, index), AluExp.const(dtype, 0));
}
/** Expression for accessing `indices` in an array recipe with variable "idx". */
function accessorAluExp(exp, st, indices) {
	const [index, valid] = st.toAluExp(indices);
	if (valid.resolve()) return exp.substitute({ idx: index });
	return AluExp.where(valid, exp.substitute({ idx: index }), AluExp.const(exp.dtype, 0));
}
function threefry2x32(k0, k1, c0, c1) {
	const rotl32 = (x, r) => (x << r | x >>> 32 - r) >>> 0;
	const ks0 = k0 >>> 0;
	const ks1 = k1 >>> 0;
	const ks2 = (ks0 ^ ks1 ^ 466688986) >>> 0;
	let x0 = c0 + ks0 >>> 0;
	let x1 = c1 + ks1 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks1 >>> 0;
	x1 = x1 + ks2 + 1 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
	x0 = x0 + ks2 >>> 0;
	x1 = x1 + ks0 + 2 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks0 >>> 0;
	x1 = x1 + ks1 + 3 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
	x0 = x0 + ks1 >>> 0;
	x1 = x1 + ks2 + 4 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks2 >>> 0;
	x1 = x1 + ks0 + 5 >>> 0;
	return [x0, x1];
}
/**
* Abramowitz & Stegun’s widely used approximation for erf(x).
*
* `erf(x) = 1 - P(t) * exp(-x^2)` for `x >= 0`, where `t = 1/(1 + p*x)` and
* `P(t) = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5`.
*
* Coefficients:
*  - p = 0.3275911
*  - a1 = 0.254829592
*  - a2 = -0.284496736
*  - a3 = 1.421413741
*  - a4 = -1.453152027
*  - a5 = 1.061405429
*
* This function computes just `E = P(t) * exp(-x^2)` for numerical reasons. The
* input is assumed to be non-negative.
*
* Reference: https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
*/
function _erfapprox$1(x) {
	const p = .3275911;
	const a1 = .254829592;
	const a2 = -.284496736;
	const a3 = 1.421413741;
	const a4 = -1.453152027;
	const a5 = 1.061405429;
	const t = 1 / (1 + p * x);
	return ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
}
function erf(x) {
	if (x >= 0) return 1 - _erfapprox$1(x);
	else return _erfapprox$1(-x) - 1;
}
function erfc(x) {
	if (x >= 0) return _erfapprox$1(x);
	else return 2 - _erfapprox$1(-x);
}
//#endregion
//#region src/routine.ts
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
var Routine = class {
	name;
	type;
	params;
	constructor(name, type, params) {
		this.name = name;
		this.type = type;
		this.params = params;
	}
};
/** One of the valid `Routine` that can be dispatched to backend. */
let Routines = /* @__PURE__ */ function(Routines) {
	/**
	* Sort along the last axis.
	*
	* This may be _unstable_ but it often doesn't matter, sorting numbers is
	* bitwise unique up to signed zeros and NaNs.
	*/
	Routines["Sort"] = "Sort";
	/** Stable sorting, returns `int32` indices and values of the sorted array. */
	Routines["Argsort"] = "Argsort";
	/**
	* Scatter updates into a zero-filled output, either replacing or adding at
	* each destination according to `ScatterOp`.
	*/
	Routines["Scatter"] = "Scatter";
	/**
	* Solve a triangular system of equations.
	*
	* The first batch of inputs `A` should be of shape `[..., N, N]` and upper
	* triangular, while the second batch `B` should be of shape `[..., M, N]`.
	*
	* Solves for `X` in the equation `A @ X.T = B.T`, where `A` is the
	* triangular matrix. This is equivalent to `X = B @ A^-T`.
	*/
	Routines["TriangularSolve"] = "TriangularSolve";
	/**
	* Cholesky decomposition of 2D positive semi-definite matrices.
	*
	* The input batch should be of shape `[..., N, N]`, and the output batch is
	* of the same shape, containing the lower-triangular matrix `L` such that
	* `A = L @ L.T`. Behavior is unspecified if A is not positive semi-definite.
	*/
	Routines["Cholesky"] = "Cholesky";
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
	Routines["LU"] = "LU";
	/**
	* Symmetric eigendecomposition using cyclic Jacobi rotations.
	*
	* The input is a batch of real symmetric matrices with shape `[..., N, N]`;
	* only the lower triangle is read. The output is a tuple `Diagonalized,
	* Vectors`, both shape `[..., N, N]`. `Diagonalized` is approximately
	* diagonal and `Vectors` contains the accumulated eigenvectors as columns.
	* Sorting eigenpairs is handled by the frontend wrapper.
	*/
	Routines["JacobiEigh"] = "JacobiEigh";
	/**
	* Complex FFT along the last axis for real/imaginary array pairs.
	*
	* Inputs and outputs have identical shape, and the backend receives a
	* factorization of the FFT dimension.
	*/
	Routines["Fft"] = "Fft";
	return Routines;
}({});
/** Operation applied when scattering an update into its destination. */
let ScatterOp = /* @__PURE__ */ function(ScatterOp) {
	ScatterOp["Update"] = "update";
	ScatterOp["Add"] = "add";
	return ScatterOp;
}({});
function runCpuRoutine(routine, inputs, outputs) {
	const { name, type } = routine;
	const inputAr = inputs.map((buf, i) => dtypedArray(type.inputDtypes[i], buf));
	const outputAr = outputs.map((buf, i) => dtypedArray(type.outputDtypes[i], buf));
	switch (name) {
		case "Sort": return runSort(type, inputAr, outputAr);
		case "Argsort": return runArgsort(type, inputAr, outputAr);
		case "Scatter": return runScatter(type, inputAr, outputAr, routine.params);
		case "TriangularSolve": return runTriangularSolve(type, inputAr, outputAr, routine.params);
		case "Cholesky": return runCholesky(type, inputAr, outputAr);
		case "LU": return runLU(type, inputAr, outputAr);
		case "JacobiEigh": return runJacobiEigh(type, inputAr, outputAr, routine.params);
		case "Fft": return runFft(type, inputAr, outputAr, routine.params);
		default:
	}
}
/** JS source for running CPU routines inside a Web Worker. */
function cpuRoutineJSForWorkers() {
	return `
const DType = ${JSON.stringify(DType)};
const Routines = ${JSON.stringify(Routines)};
const ${byteWidth.name} = ${byteWidth.toString()};
const ${isFloatDtype.name} = ${isFloatDtype.toString()};
const ${dtypedArray.name} = ${dtypedArray.toString()};
const ScatterOp = ${JSON.stringify(ScatterOp)};
${runCpuRoutine.toString()}
${runSort.toString()}
${runArgsort.toString()}
${shapeStrides.toString()}
${runScatter.toString()}
${runTriangularSolve.toString()}
${runCholesky.toString()}
${runLU.toString()}
${runJacobiEigh.toString()}
${runFft.toString()}
const __minify_safe_runCpuRoutine = ${runCpuRoutine.name};
`;
}
function runSort(type, [x], [y]) {
	const xs = type.inputShapes[0];
	if (xs.length === 0) throw new Error("sort: cannot sort a scalar");
	const n = xs[xs.length - 1];
	y.set(x);
	for (let i = 0; i < y.length; i += n) y.subarray(i, i + n).sort();
}
function runArgsort(type, [x], [y, yi]) {
	const xs = type.inputShapes[0];
	if (xs.length === 0) throw new Error("argsort: cannot sort a scalar");
	const n = xs[xs.length - 1];
	for (let offset = 0; offset < y.length; offset += n) {
		const ar = x.subarray(offset, offset + n);
		const out = y.subarray(offset, offset + n);
		const outi = yi.subarray(offset, offset + n);
		for (let i = 0; i < n; i++) outi[i] = i;
		outi.sort((a, b) => {
			const x = ar[a];
			const y = ar[b];
			if (isNaN(x)) return isNaN(y) ? 0 : 1;
			if (isNaN(y)) return -1;
			return x === y ? 0 : x < y ? -1 : 1;
		});
		for (let i = 0; i < n; i++) out[i] = ar[outi[i]];
	}
}
function shapeStrides(shape) {
	const strides = new Array(shape.length);
	let stride = 1;
	for (let i = shape.length - 1; i >= 0; i--) {
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
}
function runScatter(type, [updates, ...indices], [output], { op, shape: outputShape, axis: indexedAxes, outDim, uniqueIndices }) {
	output.fill(0);
	const accumulate = op !== "update" && !uniqueIndices;
	const updateShape = type.inputShapes[0];
	const indexShapes = type.inputShapes.slice(1);
	const indexRank = Math.max(...indexShapes.map((s) => s.length));
	const updateStrides = shapeStrides(updateShape);
	const outputStrides = shapeStrides(outputShape);
	const indexStrides = indexShapes.map(shapeStrides);
	const outputAxisToIndex = outputShape.map((_, d) => indexedAxes.indexOf(d));
	const freeUpdateDims = updateShape.map((_, i) => i).filter((i) => i < outDim || i >= outDim + indexRank);
	for (let i = 0; i < updates.length; i++) {
		let outputIndex = 0;
		let valid = true;
		let freeDim = 0;
		for (let outputAxis = 0; outputAxis < outputShape.length; outputAxis++) {
			const indexInput = outputAxisToIndex[outputAxis];
			let coord;
			if (indexInput === -1) {
				const updateDim = freeUpdateDims[freeDim++];
				coord = Math.floor(i / updateStrides[updateDim]) % updateShape[updateDim];
			} else {
				const indexShape = indexShapes[indexInput];
				const firstUpdateDim = outDim + indexRank - indexShape.length;
				let indexOffset = 0;
				for (let j = 0; j < indexShape.length; j++) {
					if (indexShape[j] === 1) continue;
					const updateDim = firstUpdateDim + j;
					const updateCoord = Math.floor(i / updateStrides[updateDim]) % updateShape[updateDim];
					indexOffset += updateCoord * indexStrides[indexInput][j];
				}
				coord = indices[indexInput][indexOffset];
			}
			if (coord < 0 || coord >= outputShape[outputAxis]) {
				valid = false;
				break;
			}
			outputIndex += coord * outputStrides[outputAxis];
		}
		if (!valid) continue;
		if (accumulate && type.inputDtypes[0] === "bool") output[outputIndex] |= updates[i];
		else if (accumulate) output[outputIndex] += updates[i];
		else output[outputIndex] = updates[i];
	}
}
function runTriangularSolve(type, [a, b], [x], { unitDiagonal }) {
	const as = type.inputShapes[0];
	const bs = type.inputShapes[1];
	if (as.length < 2) throw new Error(`triangular_solve: a must be at least 2D, got ${as}`);
	if (bs.length < 2) throw new Error(`triangular_solve: b must be at least 2D, got ${bs}`);
	const n = as[as.length - 2];
	if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
	const batch = bs[bs.length - 2];
	for (let counter = 0; counter < a.length / (n * n); counter++) {
		const a1 = a.subarray(counter * n * n, (counter + 1) * n * n);
		for (let t = 0; t < batch; t++) {
			const b1 = b.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
			const x1 = x.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
			for (let i = n - 1; i >= 0; i--) {
				let sum = b1[i];
				for (let j = i + 1; j < n; j++) sum -= a1[i * n + j] * x1[j];
				x1[i] = unitDiagonal ? sum : sum / a1[i * n + i];
			}
		}
	}
}
function runCholesky(type, [x], [y]) {
	const xs = type.inputShapes[0];
	if (xs.length < 2) throw new Error("cholesky: input must be at least 2D");
	const n = xs[xs.length - 2];
	const m = xs[xs.length - 1];
	if (n !== m) throw new Error(`cholesky: input must be square, got [${n}, ${m}]`);
	y.fill(0);
	for (let offset = 0; offset < y.length; offset += n * n) {
		const ar = x.subarray(offset, offset + n * n);
		const out = y.subarray(offset, offset + n * n);
		for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
			let sum = ar[i * n + j];
			for (let k = 0; k < j; k++) sum -= out[i * n + k] * out[j * n + k];
			out[i * n + j] = i === j ? Math.sqrt(sum) : sum / out[j * n + j];
		}
	}
}
function runLU(type, [a], [lu, pivots, perm]) {
	const shape = type.inputShapes[0];
	if (shape.length < 2) throw new Error("lu: input must be at least 2D");
	const m = shape[shape.length - 2];
	const n = shape[shape.length - 1];
	const r = Math.min(m, n);
	for (let offset = 0; offset < a.length; offset += m * n) {
		const ar = a.subarray(offset, offset + m * n);
		const out = lu.subarray(offset, offset + m * n);
		const batchIdx = offset / (m * n);
		const piv = pivots.subarray(batchIdx * r, (batchIdx + 1) * r);
		const p = perm.subarray(batchIdx * m, (batchIdx + 1) * m);
		out.set(ar);
		for (let i = 0; i < m; i++) p[i] = i;
		for (let j = 0; j < r; j++) {
			let maxVal = Math.abs(out[j * n + j]);
			let maxRow = j;
			for (let i = j + 1; i < m; i++) {
				const val = Math.abs(out[i * n + j]);
				if (val > maxVal) {
					maxVal = val;
					maxRow = i;
				}
			}
			piv[j] = maxRow;
			if (maxRow !== j) {
				for (let col = 0; col < n; col++) {
					const tmp = out[j * n + col];
					out[j * n + col] = out[maxRow * n + col];
					out[maxRow * n + col] = tmp;
				}
				const tmpP = p[j];
				p[j] = p[maxRow];
				p[maxRow] = tmpP;
			}
			const diag = out[j * n + j];
			if (diag !== 0) for (let i = j + 1; i < m; i++) {
				const factor = out[i * n + j] / diag;
				out[i * n + j] = factor;
				for (let col = j + 1; col < n; col++) out[i * n + col] -= factor * out[j * n + col];
			}
		}
	}
}
function runJacobiEigh(type, [input], [diagonalized, vectors], { maxSweeps, tolerance }) {
	const shape = type.inputShapes[0];
	if (shape.length < 2) throw new Error("jacobi_eigh: input must be at least 2D");
	const n = shape[shape.length - 1];
	if (n !== shape[shape.length - 2]) throw new Error(`jacobi_eigh: input must be square, got ${shape}`);
	if (!isFloatDtype(type.inputDtypes[0])) throw new TypeError(`jacobi_eigh: input must be floating-point`);
	function symIndex(i, j) {
		return i >= j ? i * n + j : j * n + i;
	}
	function maxAbsMatrix(a) {
		let maxAbs = 1;
		for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
		return maxAbs;
	}
	function maxAbsOffDiagonal(a) {
		let maxAbs = 0;
		for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
		return maxAbs;
	}
	function applyJacobiRotation(a, v, p, q) {
		const pp = p * n + p;
		const qq = q * n + q;
		const pq = symIndex(p, q);
		const app = a[pp];
		const aqq = a[qq];
		const apq = a[pq];
		if (apq === 0) return;
		const tau = (aqq - app) / (2 * apq);
		const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(tau * tau + 1));
		const c = 1 / Math.sqrt(t * t + 1);
		const s = t * c;
		for (let k = 0; k < n; k++) {
			if (k === p || k === q) continue;
			const kp = symIndex(k, p);
			const kq = symIndex(k, q);
			const akp = a[kp];
			const akq = a[kq];
			const nextKp = c * akp - s * akq;
			const nextKq = s * akp + c * akq;
			a[kp] = nextKp;
			a[kq] = nextKq;
		}
		a[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
		a[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
		a[pq] = 0;
		for (let k = 0; k < n; k++) {
			const kp = k * n + p;
			const kq = k * n + q;
			const vkp = v[kp];
			const vkq = v[kq];
			v[kp] = c * vkp - s * vkq;
			v[kq] = s * vkp + c * vkq;
		}
	}
	const matrixSize = n * n;
	for (let offset = 0; offset < input.length; offset += matrixSize) {
		const x = input.subarray(offset, offset + matrixSize);
		const a = diagonalized.subarray(offset, offset + matrixSize);
		const v = vectors.subarray(offset, offset + matrixSize);
		a.fill(0);
		for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) a[i * n + j] = x[i * n + j];
		v.fill(0);
		for (let i = 0; i < n; i++) v[i * n + i] = 1;
		const threshold = tolerance * maxAbsMatrix(a);
		let maxOffDiagonal = maxAbsOffDiagonal(a);
		for (let sweep = 0; sweep < maxSweeps && maxOffDiagonal > threshold; sweep++) {
			for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) applyJacobiRotation(a, v, p, q);
			maxOffDiagonal = maxAbsOffDiagonal(a);
		}
	}
}
function runFft(type, [real, imag], [outReal, outImag], { factors, inverse }) {
	const shape = type.inputShapes[0];
	if (shape.length < 1) throw new Error("fft: input must be at least 1D");
	const n = shape[shape.length - 1];
	if (n < 1) throw new Error(`fft: final axis must be non-empty, got ${n}`);
	if (type.inputDtypes[0] !== type.inputDtypes[1] || !isFloatDtype(type.inputDtypes[0])) throw new Error("fft: expected matching floating-point real/imag arrays");
	const angleScale = (inverse ? 2 : -2) * Math.PI;
	const permutation = new Uint32Array(n);
	for (let index = 0; index < n; index++) {
		let remaining = index;
		let stride = 1;
		let reversed = 0;
		for (const factor of factors) {
			const digit = remaining % factor;
			remaining = Math.floor(remaining / factor);
			stride *= factor;
			reversed += digit * (n / stride);
		}
		permutation[index] = reversed;
	}
	const scratchReal = new Array(Math.max(1, ...factors));
	const scratchImag = new Array(scratchReal.length);
	for (let offset = 0; offset < real.length; offset += n) {
		for (let index = 0; index < n; index++) {
			const source = offset + permutation[index];
			outReal[offset + index] = real[source];
			outImag[offset + index] = imag[source];
		}
		let prev = 1;
		for (let stage = 0; stage < factors.length; stage++) {
			const radix = factors[stage];
			const span = prev * radix;
			const stageScale = inverse && stage === factors.length - 1 ? 1 / n : 1;
			for (let group = 0; group < n; group += span) for (let j = 0; j < prev; j++) {
				for (let q = 0; q < radix; q++) {
					const idx = offset + group + j + q * prev;
					const angle = angleScale * q * j / span;
					const c = Math.cos(angle);
					const s = Math.sin(angle);
					const xr = outReal[idx];
					const xi = outImag[idx];
					scratchReal[q] = xr * c - xi * s;
					scratchImag[q] = xr * s + xi * c;
				}
				for (let p = 0; p < radix; p++) {
					let sumReal = 0;
					let sumImag = 0;
					for (let q = 0; q < radix; q++) {
						const angle = angleScale * q * p / radix;
						const c = Math.cos(angle);
						const s = Math.sin(angle);
						const xr = scratchReal[q];
						const xi = scratchImag[q];
						sumReal += xr * c - xi * s;
						sumImag += xr * s + xi * c;
					}
					const idx = offset + group + j + p * prev;
					outReal[idx] = sumReal * stageScale;
					outImag[idx] = sumImag * stageScale;
				}
			}
			prev = span;
		}
	}
}
//#endregion
//#region src/shape.ts
/**
* @file Lazy shape tracking for multidimensional tensors.
*
* This module provides an immutable `View` class that can be used to calculate
* shapes of arrays as operations are applied to them, lazily.
*
* Some operations like reshape() may not be representable with a single view,
* for instance, because composing reshape() with shrink() leads to a
* non-contiguous range of memory locations. This is why `ShapeTracker` is a
* list of views.
*
* Indexing into a `ShapeTracker` or `View` can be folded into shader code.
*
* Originally based on tinygrad's implementation of shape tracking in the
* `tinygrad.shape` module. But this version is simplified a bit. I'm not really
* trying to innovate on shape tracking in this library, so if I have doubts on
* something, it'll just be copied from tinygrad (with comments).
*
* This file is a bit longer than the original, since Python is more concise.
*/
const jstr = JSON.stringify;
/** Remove "1" dimensions from the strides list. */
function canonicalizeStrides(shape, strides) {
	const newStrides = [];
	for (let i = 0; i < shape.length; i++) if (shape[i] === 1) newStrides.push(0);
	else newStrides.push(strides[i]);
	return newStrides;
}
/** Get the strides for a shape in default row-major order. */
function defaultStrides(shape) {
	if (shape.length === 0) return [];
	const strides = rep(shape.length, 1);
	for (let i = shape.length - 1; i > 0; i--) strides[i - 1] = shape[i] * strides[i];
	return canonicalizeStrides(shape, strides);
}
/** Merge contiguous subparts or zero-strided dimensions in a view. */
function mergeDims(shape, strides, mask) {
	if (shape.length === 0) return [];
	if (shape.length !== strides.length || mask && shape.length !== mask.length) throw new Error("internal: invalid args to mergeDims");
	const ret = [[
		shape[0],
		strides[0],
		strides[0] !== 0 ? shape[0] : 0
	]];
	let merging = mask ? mask[0][1] - mask[0][0] === 1 : shape[0] === 1;
	for (let i = 1; i < shape.length; i++) {
		const [s, st] = [shape[i], strides[i]];
		if (s === 1) continue;
		const [lastS, lastSt, lastPreExpandS] = ret[ret.length - 1];
		if (merging || lastSt === s * st) ret[ret.length - 1] = [
			lastS * s,
			st,
			merging ? s : lastPreExpandS * s
		];
		else ret.push([
			s,
			st,
			s
		]);
		merging = mask ? mask[i][1] - mask[i][0] === 1 : false;
	}
	return ret;
}
/** Return the new mask if a reshape if possible, otherwise `null`. */
function reshapeMask(maskInput, oldShape, newShape) {
	const newMask = [];
	let rMasksI = maskInput.length;
	let rShapeI = oldShape.length;
	let rNewShapeI = newShape.length;
	const rMasks = () => rMasksI ? maskInput[--rMasksI] : [0, 1];
	const rShape = () => rShapeI ? oldShape[--rShapeI] : 1;
	const rNewShape = () => rNewShapeI ? newShape[--rNewShapeI] : 1;
	let currStride = 1;
	let [oldDim, newDim, mask] = [
		rShape(),
		rNewShape(),
		rMasks()
	];
	while (newMask.length < newShape.length) {
		const [l, r] = mask;
		const nextStride = newDim * currStride;
		if (oldDim === nextStride) {
			newMask.push([intdiv(l, currStride), intdiv(r - 1, currStride) + 1]);
			currStride = 1;
			[oldDim, newDim, mask] = [
				rShape(),
				rNewShape(),
				rMasks()
			];
		} else if (oldDim > nextStride) {
			if (oldDim % nextStride !== 0) return null;
			if ((l % nextStride !== 0 || r % nextStride !== 0) && intdiv(l, nextStride) !== intdiv(r - 1, nextStride)) return null;
			newMask.push([intdiv(l % nextStride, currStride), intdiv((r - 1) % nextStride, currStride) + 1]);
			[currStride, newDim] = [nextStride, rNewShape()];
		} else {
			const nextMask = rMasks();
			if (!deepEqual(mask, [0, oldDim]) && l !== r && nextMask[1] - nextMask[0] !== 1) return null;
			mask = [nextMask[0] * oldDim + l, (nextMask[1] - 1) * oldDim + r];
			oldDim *= rShape();
		}
	}
	return newMask.reverse();
}
/**
* A multidimensional view into memory. An array can be thought of as the
* combination of a linear buffer of memory, along with a `View`.
*
* Formula for getting a data point is basically:
*   1. Check if ∀i. 0 <= dim[i] < shape[i], otherwise out of bounds.
*   2. If mask exists, and ∃i. dim[i] ∉ mask[i], return 0.
*   2. Otherwise, look at this memory address: offset + ∑(strides[i] * dim[i]).
*/
var View = class View {
	shape;
	strides;
	offset;
	mask;
	#size;
	#contiguous;
	constructor(shape, strides, offset, mask) {
		this.shape = shape;
		this.strides = strides;
		this.offset = offset;
		this.mask = mask;
	}
	static create(shape, strides, offset = 0, mask = null) {
		if (shape.some((s) => s < 0)) throw new Error("View shape must be non-negative");
		strides = strides ? canonicalizeStrides(shape, strides) : defaultStrides(shape);
		if (shape.includes(0)) return new View(shape, rep(shape.length, 0), 0, null);
		if (mask !== null && mask.every(([b, e], i) => b === 0 && e === shape[i])) mask = null;
		if (mask !== null) {
			const elimDims = [];
			let hasNoData = false;
			for (let i = 0; i < shape.length; i++) {
				const [b, e] = mask[i];
				if (b + 1 >= e) elimDims.push(i);
				if (b >= e) hasNoData = true;
			}
			if (elimDims.length) {
				if (hasNoData) {
					strides = rep(shape.length, 0);
					offset = 0;
					mask = rep(shape.length, () => [0, 0]);
				}
				for (const i of elimDims) {
					offset += strides[i] * mask[i][0];
					strides[i] = 0;
				}
			}
		}
		return new View(shape, strides, offset, mask);
	}
	get ndim() {
		return this.shape.length;
	}
	get size() {
		if (this.#size === void 0) this.#size = prod(this.shape);
		return this.#size;
	}
	/** Whether this is a default, contiguous, unaltered view of the data (identity). */
	get contiguous() {
		if (this.#contiguous === void 0) this.#contiguous = this.size === 0 || this.offset === 0 && this.mask === null && deepEqual(this.strides, defaultStrides(this.shape));
		return this.#contiguous;
	}
	/** Return the range of data being indexed in this view, or [0, 0] if none. */
	dataRange() {
		if (this.size === 0 || this.mask && this.mask[0][0] === this.mask[0][1]) return [0, 0];
		let min = this.offset;
		let max = this.offset;
		for (let i = 0; i < this.ndim; i++) {
			let [lo, hi] = this.mask ? this.mask[i] : [0, this.shape[i]];
			--hi;
			const s = this.strides[i];
			if (s > 0) {
				min += s * lo;
				max += s * hi;
			} else if (s < 0) {
				min += s * hi;
				max += s * lo;
			}
		}
		return [min, max + 1];
	}
	/** Produce an AluExp for evaluating this view at an index. */
	toAluExp(idxs) {
		let iexpr = AluExp.i32(this.offset);
		let vexpr = AluExp.bool(true);
		for (let i = this.ndim - 1; i >= 0; i--) {
			const idx = idxs[i];
			if (this.shape[i] !== 1 && this.strides[i] !== 0) iexpr = AluExp.add(AluExp.mul(idx, AluExp.i32(this.strides[i])), iexpr);
			if (this.mask) {
				if (this.mask[i][0] !== 0) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][0])).not(), vexpr);
				if (this.mask[i][1] !== this.shape[i]) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][1])), vexpr);
			}
		}
		return [iexpr, vexpr];
	}
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
	compose(v1) {
		const v2 = this;
		if (v2.contiguous) return v1;
		if (v1.contiguous) {
			if (deepEqual(v1.shape, v2.shape)) return v2;
			if (v1.size === v2.size) {
				const ret = v2.reshape(v1.shape);
				if (ret !== null) return ret;
			}
		}
		if (v1.mask !== null) {
			const newV1 = v1.shrink(v1.mask);
			const merged = v2.compose(newV1);
			return merged ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]])) : null;
		}
		const origin = unravel(v2.shape, v1.offset);
		const terms = rep(v2.ndim, () => []);
		const strides = rep(v1.ndim, 0);
		for (let d1 = 0; d1 < v1.strides.length; d1++) {
			const st = v1.strides[d1];
			if (st === 0) continue;
			const unravelOffset = unravel(v2.shape, v1.offset + st);
			for (let d2 = 0; d2 < v2.ndim; d2++) {
				const o = origin[d2];
				const diff = unravelOffset[d2] - o;
				if (diff === 0) continue;
				terms[d2].push([d1, diff]);
				strides[d1] += diff * v2.strides[d2];
			}
		}
		let [mergedSize, mergedTermMin, mergedTermMax] = [
			1,
			0,
			0
		];
		const extents = [];
		for (let i = v2.ndim - 1; i >= 0; i--) {
			const term = terms[i];
			const s = v2.shape[i];
			let [tmin, tmax] = [origin[i], origin[i]];
			for (const [d1, s1] of term) if (s1 > 0) tmax += (v1.shape[d1] - 1) * s1;
			else if (s1 < 0) tmin += (v1.shape[d1] - 1) * s1;
			mergedTermMin += tmin * mergedSize;
			mergedTermMax += tmax * mergedSize;
			mergedSize *= s;
			if (mergedTermMin >= 0 && mergedTermMax < mergedSize) {
				extents.push([
					mergedSize,
					mergedTermMin,
					mergedTermMax
				]);
				[mergedSize, mergedTermMin, mergedTermMax] = [
					1,
					0,
					0
				];
			}
		}
		if (mergedTermMin !== 0 || mergedTermMax !== 0) return null;
		extents.reverse();
		const v2Shape = extents.map(([s]) => s);
		if (!deepEqual(v2Shape, v2.shape)) {
			const reshapedV2 = v2.reshape(v2Shape);
			if (reshapedV2 === null) return null;
			if (!deepEqual(reshapedV2.shape, v2.shape)) return reshapedV2.compose(v1);
		}
		if (v2.mask !== null) {
			const newB = rep(v1.ndim, 0);
			const newE = v1.shape.slice();
			let bad = false;
			for (let d2 = 0; d2 < v2.ndim; d2++) {
				const [b, e] = v2.mask[d2];
				const o = origin[d2];
				const term = terms[d2];
				const [_, tmin, tmax] = extents[d2];
				if (b <= tmin && tmax < e) continue;
				if (term.length !== 1) if (term.length === 0 && newE.length) newE[0] = 0;
				else bad = true;
				else {
					const [d1, s1] = term[0];
					newB[d1] = Math.max(newB[d1], Math.ceil((s1 > 0 ? b - o : e - o - 1) / s1));
					newE[d1] = Math.min(newE[d1], Math.floor((s1 < 0 ? b - o : e - o - 1) / s1) + 1);
				}
			}
			for (let d1 = 0; d1 < v1.ndim; d1++) if (newB[d1] !== 0 || newE[d1] !== v1.shape[d1]) return v2.compose(View.create(v1.shape, v1.strides, v1.offset, zip(newB, newE)));
			if (bad) return null;
		}
		let finalOffset = v2.offset;
		for (let d2 = 0; d2 < v2.ndim; d2++) finalOffset += origin[d2] * v2.strides[d2];
		return View.create(v1.shape, strides, finalOffset, null);
	}
	/** Attempt to simplify this view into a smaller reshaped form. */
	minify() {
		const minShape = mergeDims(this.shape, this.strides, this.mask).map((x) => x[0]);
		const nv = this.reshape(minShape);
		return nv ? nv : this;
	}
	/** Pad the view with zeros on each dimension. */
	pad(arg) {
		if (arg.length !== this.ndim || !arg.every(([b, e]) => b >= 0 && e >= 0)) throw new Error(`invalid pad ${jstr(arg)} for ${jstr(this.shape)}`);
		if (arg.every(([b, e]) => b === 0 && e === 0)) return this;
		const zvarg = arg.map(([b, e], i) => [-b, this.shape[i] + e]);
		const mask = arg.map(([b, _e], i) => [b, this.shape[i] + b]);
		return this.#unsafeResize(zvarg, mask);
	}
	/** Shrink the view by taking a subarray. */
	shrink(arg) {
		if (arg.length !== this.ndim || !arg.every(([b, e], i) => 0 <= b && b <= e && e <= this.shape[i])) throw new Error(`invalid shrink ${jstr(arg)} for ${jstr(this.shape)}`);
		return this.#unsafeResize(arg);
	}
	#unsafeResize(arg, mask) {
		const offset = this.strides.map((s, i) => s * arg[i][0]).reduce((a, b) => a + b, 0);
		if (this.mask) {
			const nmask = this.mask.map(([mx, my], i) => [Math.max(0, Math.min(mx - arg[i][0], arg[i][1] - arg[i][0])), Math.max(0, Math.min(my - arg[i][0], arg[i][1] - arg[i][0]))]);
			mask = mask ? mask.map(([mx, my], i) => [Math.max(mx, nmask[i][0]), Math.min(my, nmask[i][1])]) : nmask;
		}
		return View.create(arg.map(([b, e]) => e - b), this.strides, this.offset + offset, mask);
	}
	/** Expand one or more axes with length "1" by repeating the data. */
	expand(newShape) {
		if (newShape.length !== this.ndim) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
		for (let i = 0; i < this.ndim; i++) if (newShape[i] !== this.shape[i] && this.shape[i] !== 1) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
		if (this.size === 0) return View.create(newShape);
		const mask = this.mask ? this.mask.map((m, i) => this.shape[i] === newShape[i] ? m : m[0] === 0 && m[1] === 1 ? [0, newShape[i]] : [0, 0]) : null;
		return View.create(newShape, this.strides, this.offset, mask);
	}
	/** Permute the axes of an array. */
	permute(axis) {
		if (!isPermutation(axis, this.ndim)) throw new Error(`Invalid permutation ${jstr(axis)} of len ${this.ndim}`);
		const newShape = axis.map((a) => this.shape[a]);
		const newStrides = axis.map((a) => this.strides[a]);
		const newMask = this.mask ? axis.map((a) => this.mask[a]) : null;
		return View.create(newShape, newStrides, this.offset, newMask);
	}
	/** Flip (reverse) one or more axes of the view. */
	flip(arg) {
		if (arg.length !== this.ndim) throw new Error(`Invalid flip ${jstr(arg)} for ${jstr(this.shape)}`);
		const strides = this.strides.slice();
		let offset = this.offset;
		const mask = this.mask ? this.mask.slice() : null;
		for (let i = 0; i < this.ndim; i++) {
			const s = this.shape[i];
			if (arg[i]) {
				strides[i] = -strides[i];
				offset += (s - 1) * this.strides[i];
				if (mask) mask[i] = [s - mask[i][1], s - mask[i][0]];
			}
		}
		return View.create(this.shape, strides, offset, mask);
	}
	/** Reshape the view into a new shape. */
	reshape(newShape) {
		if (deepEqual(this.shape, newShape)) return this;
		if (newShape.some((s) => s < 0)) throw new Error(`Reshape cannot have negative numbers ${jstr(newShape)}`);
		if (this.size !== prod(newShape)) throw new Error(`Reshape size ${jstr(this.shape)} -> ${jstr(newShape)}`);
		if (this.size === 0) return View.create(newShape);
		if (newShape.length === 0 && this.mask?.some(([b, e]) => b === e)) return null;
		if (this.contiguous) return View.create(newShape);
		const rStrides = [];
		const merge = mergeDims(this.shape, this.strides, this.mask);
		let rShapeIdx = newShape.length;
		for (let i = merge.length - 1; i >= 0; i--) {
			let [mergedSize, newStride, realSize] = merge[i];
			let acc = 1;
			while (acc < mergedSize && rShapeIdx > 0) {
				const newDim = newShape[--rShapeIdx];
				rStrides.push(newStride * acc);
				acc *= newDim;
				if (acc >= realSize) newStride = 0;
			}
			if (acc !== mergedSize) return null;
		}
		const newStrides = rep(newShape.length - rStrides.length, 0).concat(rStrides.reverse());
		if (!this.mask) return View.create(newShape, newStrides, this.offset);
		const newMask = reshapeMask(this.mask, this.shape, newShape);
		if (!newMask) return null;
		let newOffset = this.offset;
		for (let i = 0; i < this.ndim; i++) newOffset += this.strides[i] * this.mask[i][0];
		for (let i = 0; i < newShape.length; i++) newOffset -= newStrides[i] * newMask[i][0];
		return View.create(newShape, newStrides, newOffset, newMask);
	}
};
/**
* Find position of `offset` in each dimension within an existing shape. Like
* `numpy.unravel_index` in behavior.
*/
function unravel(shape, offset) {
	let acc = 1;
	const idxs = [];
	for (let i = shape.length - 1; i >= 0; i--) {
		const d = shape[i];
		idxs.push(Math.floor(offset / acc) % d);
		acc *= d;
	}
	return idxs.reverse();
}
/** Generate a list of AluExp for computing unravel(). */
function unravelAlu(shape, offset) {
	let acc = 1;
	const idxs = [];
	for (let i = shape.length - 1; i >= 0; i--) {
		const d = shape[i];
		idxs.push(AluExp.mod(AluExp.idiv(offset, AluExp.i32(acc)), AluExp.i32(d)));
		acc *= d;
	}
	return idxs.reverse();
}
/**
* Array shape after applying movement operations, as a series of views.
*
* Each view is applied, then treated as if it were a contiguous array of its
* shape, then used as the virtual buffer for the next view.
*/
var ShapeTracker = class ShapeTracker {
	views;
	constructor(views) {
		this.views = views;
	}
	/** Compose this shape tracker with another, applying it after this one. */
	compose(other) {
		if (this.contiguous) return other;
		let ret = this;
		for (const v of other.views) ret = new ShapeTracker(ret.views.concat(v)).simplify();
		return ret;
	}
	static fromShape(shape) {
		return new ShapeTracker([View.create(shape)]);
	}
	get contiguous() {
		return this.views.length === 1 && this.views[0].contiguous;
	}
	get consecutive() {
		return this.views.length === 1 && this.views[0].mask === null && deepEqual(this.views[0].strides, defaultStrides(this.views[0].shape));
	}
	get lastStrides() {
		return this.views[this.views.length - 1].strides;
	}
	get shape() {
		return this.views[this.views.length - 1].shape;
	}
	get size() {
		return this.views[this.views.length - 1].size;
	}
	toAluExp(idxs) {
		let [iexpr, vexpr] = this.views[this.views.length - 1].toAluExp(idxs);
		for (let i = this.views.length - 2; i >= 0; i--) {
			const view = this.views[i].minify();
			const exprs = view.toAluExp(unravelAlu(view.shape, iexpr));
			iexpr = exprs[0];
			vexpr = AluExp.mul(vexpr, exprs[1]);
		}
		return [iexpr.simplify(), vexpr.simplify()];
	}
	simplify() {
		const views = this.views.slice();
		while (views.length >= 2) {
			const newView = views[views.length - 2].compose(views[views.length - 1]);
			if (newView === null) break;
			views.splice(views.length - 2, 2, newView);
		}
		return new ShapeTracker(views);
	}
	pad(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.pad(arg)));
	}
	shrink(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.shrink(arg)));
	}
	expand(newShape) {
		return new ShapeTracker(applyLast(this.views, (x) => x.expand(newShape)));
	}
	permute(axis) {
		return new ShapeTracker(applyLast(this.views, (x) => x.permute(axis)));
	}
	flip(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.flip(arg)));
	}
	reshape(newShape) {
		const newView = this.views[this.views.length - 1].reshape(newShape);
		return new ShapeTracker(newView === null ? this.views.concat(View.create(newShape)) : this.views.toSpliced(this.views.length - 1, 1, newView));
	}
	/** Broadcast along the given new axes, then expand the shape. */
	broadcast(newShape, axis) {
		let st = this;
		if (axis.length > 0) {
			const unsqueezed = [...st.shape];
			for (const i of sorted(axis)) unsqueezed.splice(i, 0, 1);
			st = st.reshape(unsqueezed);
		}
		return st.expand(newShape);
	}
	/**
	* Repeat data in each axis by a positive number of repetitions.
	*
	* - If `tile` is true (default): [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
	* - If `tile` is false: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].
	*/
	repeat(reps, tile = true) {
		if (reps.length > this.shape.length) throw new Error(`Too many repeats ${jstr(reps)} for shape ${jstr(this.shape)}`);
		if (reps.some((c) => c <= 0)) throw new Error(`Invalid repeats ${jstr(reps)}`);
		if (reps.length === 0) return this;
		const noop = this.shape.slice(0, -reps.length);
		const shape = this.shape.slice(-reps.length);
		return this.broadcast([...noop, ...shape.flatMap((s, i) => tile ? [reps[i], s] : [s, reps[i]])], shape.map((_, i) => noop.length + 2 * i + (tile ? 0 : 1))).reshape([...noop, ...shape.map((s, i) => s * reps[i])]);
	}
	/** Move axis i to axis j. */
	moveaxis(i, j) {
		const perm = range(this.shape.length);
		perm.splice(i, 1);
		perm.splice(j, 0, i);
		return this.permute(perm);
	}
	/** Like pad(), but allows for negative values. */
	padOrShrink(arg) {
		const padArg = [];
		const shrinkArg = [];
		for (let i = 0; i < arg.length; i++) {
			const [b, e] = arg[i];
			if (b < -this.shape[i] || e < -this.shape[i] || b + e < -this.shape[i]) throw new Error(`Invalid padOrShrink ${jstr(arg)} for ${jstr(this.shape)}`);
			padArg.push([Math.max(0, b), Math.max(0, e)]);
			shrinkArg.push([Math.max(0, -b), this.shape[i] - Math.max(0, -e)]);
		}
		return this.shrink(shrinkArg).pad(padArg);
	}
};
function applyLast(ar, f) {
	return ar.toSpliced(ar.length - 1, 1, f(ar[ar.length - 1]));
}
//#endregion
//#region src/tuner.ts
/**
* @file Optimizations applied to kernels by different backends.
*
* The main optimizations (for reductions) are:
*
* - "Upcast": Multiple values are computed per thread, along a non-reduction
*   dimension. If appropriate, this lowers to vector/SIMD instructions. Each
*   thread computes a chunk of output values, which helps with cache
*   performance (e.g., matmul tiling).
*
* - "Unroll": Similar to Upcast, but along a loop dimension, which translates
*   to loop unrolling. You increment the loop index by the unroll factor. This
*   does not use vector/SIMD instructions.
*
* - "Group": Multiple threads compute the same value. For example, when summing
*   up the numbers in a vector, K threads each accumulate 1/K of the vector,
*   stores in shared memory, and thread 0 accumulates at the end.
*   - Regular order: 4 threads grouped as [1234123412341234]
*   - "Top": 4 threads grouped as [1111222233334444]
*
* These are inspired by Tinygrad's heuristic optimizations.
* https://github.com/tinygrad/tinygrad/blob/685d5c46df/tinygrad/codegen/heuristic.py
*/
/** Stores dimensions of the kernel's applied shape. Globals start at 0. */
var TuneDims = class {
	st;
	outputSt;
	groups;
	reduce;
	unroll;
	upcast;
	get end() {
		return this.st.shape.length;
	}
	constructor(shape) {
		this.st = ShapeTracker.fromShape(shape);
		this.outputSt = ShapeTracker.fromShape(shape.slice(0, -1));
		this.groups = this.st.shape.length - 1;
		this.reduce = this.st.shape.length - 1;
		this.unroll = this.st.shape.length;
		this.upcast = this.st.shape.length;
	}
	applyLocal(axis, amount) {
		if (axis >= this.groups) throw new Error("Cannot localize reduction axis");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Localize by ${amount} on axis length ${length}`);
		if (length !== amount) {
			this.groups++, this.reduce++, this.unroll++, this.upcast++;
			this.st = this.st.reshape([
				...this.st.shape.slice(0, axis),
				length / amount,
				amount,
				...this.st.shape.slice(axis + 1)
			]);
			this.outputSt = this.outputSt.reshape([
				...this.outputSt.shape.slice(0, axis),
				length / amount,
				amount,
				...this.outputSt.shape.slice(axis + 1)
			]);
			axis++;
		}
		this.st = this.st.permute([
			...range(axis),
			...range(axis + 1, this.groups),
			axis,
			...range(this.groups, this.st.shape.length)
		]);
		this.outputSt = this.outputSt.permute([
			...range(axis),
			...range(axis + 1, this.groups),
			axis,
			...range(this.groups, this.outputSt.shape.length)
		]);
	}
	applyUpcast(axis, amount) {
		if (axis >= this.groups) throw new Error("Cannot upcast along reduction axis");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Upcast by ${amount} on axis length ${length}`);
		this.st = this.st.reshape([
			...this.st.shape.slice(0, axis),
			length / amount,
			amount,
			...this.st.shape.slice(axis + 1)
		]).permute([
			...range(axis + 1),
			...range(axis + 2, this.st.shape.length + 1),
			axis + 1
		]);
		this.outputSt = this.outputSt.reshape([
			...this.outputSt.shape.slice(0, axis),
			length / amount,
			amount,
			...this.outputSt.shape.slice(axis + 1)
		]).permute([
			...range(axis + 1),
			...range(axis + 2, this.outputSt.shape.length + 1),
			axis + 1
		]);
	}
	applyUnroll(axis, amount) {
		if (axis < this.groups) throw new Error("Cannot unroll non-reduce axis");
		if (axis >= this.unroll) throw new Error("Axis already unrolled");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Unroll by ${amount} on axis length ${length}`);
		if (length === amount) {
			this.st = this.st.permute([
				...range(axis),
				...range(axis + 1, this.upcast),
				axis,
				...range(this.upcast, this.st.shape.length)
			]);
			if (axis < this.reduce) this.reduce--;
			this.unroll--;
		} else {
			this.st = this.st.reshape([
				...this.st.shape.slice(0, axis),
				length / amount,
				amount,
				...this.st.shape.slice(axis + 1)
			]).permute([
				...range(axis + 1),
				...range(axis + 2, this.upcast + 1),
				axis + 1,
				...range(this.upcast + 1, this.st.shape.length + 1)
			]);
			this.upcast++;
		}
	}
	applyGroup(axis, amount) {
		if (axis < this.reduce || axis >= this.unroll) throw new Error("Can only group reduction axes");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Group by ${amount} on axis length ${length}`);
		this.st = this.st.reshape([
			...this.st.shape.slice(0, axis),
			length / amount,
			amount,
			...this.st.shape.slice(axis + 1)
		]).permute([
			...range(this.reduce),
			axis + 1,
			...range(this.reduce, axis + 1),
			...range(axis + 2, this.st.shape.length + 1)
		]);
		this.reduce++;
		this.unroll++;
		this.upcast++;
	}
};
/** Tuning step that does not apply any optimization. */
function tuneNullopt(kernel) {
	let exp = kernel.exp;
	const vars = {};
	vars.gidx = AluExp.special("int32", "gidx", kernel.size);
	if (kernel.reduction) {
		vars.ridx = AluExp.special("int32", "ridx", kernel.reduction.size);
		if (exp.dtype !== kernel.reduction.dtype) exp = AluExp.cast(kernel.reduction.dtype, exp);
	}
	return {
		exp: exp.substitute(vars).rewriteGlobalViews().simplify(),
		epilogue: kernel.reduction?.epilogue.substitute({ gidx: vars.gidx }).rewriteGlobalViews().simplify(),
		outputIdxExp: vars.gidx,
		threadCount: kernel.size,
		size: { reduce: kernel.reduction ? kernel.reduction.size : 0 }
	};
}
/** Tuning for WebGPU kernels. */
function tuneWebgpu(kernel) {
	const reduction = kernel.reduction;
	if (!reduction) return tuneNullopt(kernel);
	const exp = AluExp.cast(reduction.dtype, kernel.exp);
	if (exp.collect((exp) => exp.op === "GlobalIndex").length > 0) {
		if (DEBUG >= 4) console.info("Tuning: Found GlobalIndex ops, skipping opt.");
		return tuneNullopt(kernel);
	}
	const globalViews = exp.collect((exp) => exp.op === "GlobalView");
	if (globalViews.length === 0) {
		if (DEBUG >= 4) console.info("Tuning: No GlobalView ops found in kernel.");
		return tuneNullopt(kernel);
	}
	const shape = globalViews[0].arg[1].shape;
	const expectedSrc = [...unravelAlu(shape.slice(0, -1), AluVar.gidx), AluVar.ridx].map((e) => e.simplify());
	for (const gv of globalViews) if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
		if (DEBUG >= 4) console.info("Tuning: GlobalView src[] not consistent with reduction.");
		return tuneNullopt(kernel);
	}
	if (shape[shape.length - 1] !== reduction.size) throw new Error("Invariant violation: shape doesn't match reduction size.");
	const sts = globalViews.map((gv) => gv.arg[1]);
	for (const st of sts) if (!deepEqual(st.shape, shape)) throw new Error("Invariant violation: GlobalView shape mismatch");
	const dim = new TuneDims(shape);
	const upcastedAxis = /* @__PURE__ */ new Set();
	while (prod(dim.st.shape.slice(0, dim.groups)) >= 1024) {
		const choices = [];
		const composedSts = sts.map((st) => st.compose(dim.st));
		for (let axis = 0; axis < dim.groups; axis++) for (const [priority, amount] of [
			4,
			3,
			5
		].entries()) if (!upcastedAxis.has(axis) && dim.st.shape[axis] % amount === 0 && composedSts.some((st) => st.lastStrides[axis] === 0 && st.lastStrides.slice(dim.unroll).every((stride) => stride > 0))) {
			let nonzeroStrides = 0;
			let totalStrides = 0;
			for (const st of composedSts) {
				nonzeroStrides += st.lastStrides[axis] > 0 ? 1 : 0;
				totalStrides += st.lastStrides[axis];
			}
			choices.push([
				nonzeroStrides,
				totalStrides,
				axis,
				priority,
				amount
			]);
		}
		if (choices.length > 0) {
			choices.sort(lexCompare);
			dim.applyUpcast(choices[0][2], choices[0][4]);
			upcastedAxis.add(choices[0][2]);
		} else break;
	}
	const groupCandidateHasContiguousInputs = sts.map((st) => st.compose(dim.st)).every((st) => {
		return !st.lastStrides.slice(0, dim.groups).some((stride) => stride !== 0) || Math.abs(st.lastStrides[dim.reduce]) <= 1;
	});
	if (reduction.op === "Add" && reduction.dtype === "float32" && groupCandidateHasContiguousInputs && prod(dim.st.shape.slice(0, dim.groups)) < 4096 && prod(dim.st.shape.slice(dim.reduce, dim.unroll)) >= 512) {
		const axis = dim.reduce;
		let amount = 16;
		while (amount > 1 && dim.st.shape[axis] % amount !== 0) amount /= 2;
		if (amount > 1) dim.applyGroup(axis, amount);
	}
	if (!/Mobi|Android/i.test(navigator.userAgent) && dim.reduce < dim.unroll && (prod(dim.st.shape.slice(dim.unroll)) <= 4 || dim.unroll === dim.upcast && prod(dim.st.shape.slice(dim.upcast)) < 64)) {
		const s = dim.st.shape[dim.unroll - 1];
		if (0 < s && s <= 32) dim.applyUnroll(dim.reduce, s);
		else for (const splits of [4, 2]) if (s % splits === 0) {
			dim.applyUnroll(dim.unroll - 1, splits);
			break;
		}
	}
	for (const ax of sorted(upcastedAxis)) {
		const s = dim.st.shape[ax];
		for (const amount of [8, 4]) if (s % amount === 0) {
			dim.applyLocal(ax, amount);
			break;
		}
	}
	const indices = [];
	const addIndices = (s, exp) => {
		if (s.length === 0) return;
		else if (s.length === 1) indices.push(exp);
		else indices.push(...unravelAlu(s, exp));
	};
	if (0 < dim.groups) {
		const s = dim.st.shape.slice(0, dim.groups);
		addIndices(s, AluExp.special("int32", "gidx", prod(s)));
	}
	if (dim.groups < dim.reduce) {
		const s = dim.st.shape.slice(dim.groups, dim.reduce);
		addIndices(s, AluExp.special("int32", "group", prod(s)));
	}
	if (dim.reduce <= dim.unroll) {
		const s = dim.st.shape.slice(dim.reduce, dim.unroll);
		addIndices(s, AluExp.special("int32", "ridx", prod(s)));
	}
	if (dim.unroll < dim.upcast) addIndices(dim.st.shape.slice(dim.unroll, dim.upcast), AluVar.unroll);
	if (dim.upcast < dim.end) addIndices(dim.st.shape.slice(dim.upcast), AluVar.upcast);
	let newExp = exp.rewrite((exp) => {
		if (exp.op === "GlobalView") {
			const gid = exp.arg[0];
			const st = exp.arg[1];
			return accessorGlobal(exp.dtype, gid, st.compose(dim.st), indices);
		}
	});
	const [iexpr, vexpr] = dim.st.toAluExp(indices);
	if (vexpr.min !== 1) throw new Error("Invariant violation: vexpr !== true");
	newExp = newExp.substitute({
		gidx: AluExp.idiv(iexpr, AluExp.i32(reduction.size)).simplify(),
		ridx: AluExp.mod(iexpr, AluExp.i32(reduction.size)).simplify()
	});
	const outputGidx = dim.outputSt.shape.slice(0, dim.groups);
	const outputUpcast = dim.outputSt.shape.slice(dim.groups);
	const outputIndices = [...unravelAlu(outputGidx, AluExp.special("int32", "gidx", prod(outputGidx))), ...unravelAlu(outputUpcast, AluVar.upcast)];
	const [outputIdxExp, _] = dim.outputSt.toAluExp(outputIndices);
	const newEpilogue = reduction.epilogue.rewrite((exp) => {
		if (exp.op === "GlobalView") {
			const gid = exp.arg[0];
			const st = exp.arg[1];
			return accessorGlobal(exp.dtype, gid, st.compose(dim.outputSt), outputIndices);
		}
	});
	if (prod(dim.st.shape.slice(dim.groups, dim.upcast)) !== reduction.size) throw new Error(`Invariant violation: reduction size ${reduction.size} does not match tuned dims ${JSON.stringify(dim.st.shape.slice(dim.groups, dim.upcast))}`);
	const size = {
		groups: prod(dim.st.shape.slice(dim.groups, dim.reduce)),
		reduce: prod(dim.st.shape.slice(dim.reduce, dim.unroll)),
		unroll: prod(dim.st.shape.slice(dim.unroll, dim.upcast)),
		upcast: prod(dim.st.shape.slice(dim.upcast))
	};
	return {
		exp: newExp.simplify(),
		epilogue: newEpilogue.simplify(),
		outputIdxExp: outputIdxExp.simplify(),
		threadCount: kernel.size / size.upcast * size.groups,
		size
	};
}
//#endregion
//#region src/backend/cpu.ts
/** Most basic implementation of `Backend` for testing. */
var CpuBackend = class {
	type = "cpu";
	maxArgs = Infinity;
	#buffers;
	#nextSlot;
	constructor() {
		this.#buffers = /* @__PURE__ */ new Map();
		this.#nextSlot = 1;
	}
	malloc(size, initialData) {
		const buffer = new Uint8Array(size);
		if (initialData) {
			if (initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
			buffer.set(initialData);
		}
		const slot = this.#nextSlot++;
		this.#buffers.set(slot, {
			buffer,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) this.#buffers.delete(slot);
	}
	async read(slot, start, count) {
		return this.readSync(slot, start, count);
	}
	readSync(slot, start, count) {
		const buffer = this.#getBuffer(slot);
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.byteLength - start;
		return buffer.slice(start, start + count);
	}
	async prepareKernel(kernel) {
		return this.prepareKernelSync(kernel);
	}
	prepareKernelSync(kernel) {
		return new Executable(kernel, void 0);
	}
	async prepareRoutine(routine) {
		return this.prepareRoutineSync(routine);
	}
	prepareRoutineSync(routine) {
		return new Executable(routine, void 0);
	}
	dispatch(exe, inputs, outputs) {
		if (exe.source instanceof Routine) return runCpuRoutine(exe.source, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
		const kernel = exe.source;
		const { exp, epilogue } = tuneNullopt(kernel);
		const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
		const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
		const usedArgs = new Map([...exp.collect((exp) => exp.op === "GlobalIndex"), ...epilogue ? epilogue.collect((exp) => exp.op === "GlobalIndex") : []].map((exp) => [exp.arg[0], exp.dtype]));
		const inputArrays = inputBuffers.map((buf, i) => {
			const dtype = usedArgs.get(i);
			if (!dtype) return null;
			return dtypedArray(dtype, buf);
		});
		const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);
		const globals = (gid, bufidx) => {
			if (gid < 0 || gid >= inputArrays.length) throw new Error("gid out of bounds: " + gid);
			if (bufidx < 0 || bufidx >= inputArrays[gid].length) throw new Error("bufidx out of bounds: " + bufidx);
			return inputArrays[gid][bufidx];
		};
		if (!kernel.reduction) for (let i = 0; i < kernel.size; i++) outputArray[i] = exp.evaluate({ gidx: i }, globals);
		else for (let i = 0; i < kernel.size; i++) {
			let acc = kernel.reduction.identity;
			for (let j = 0; j < kernel.reduction.size; j++) {
				const item = exp.evaluate({
					gidx: i,
					ridx: j
				}, globals);
				acc = kernel.reduction.evaluate(acc, item);
			}
			outputArray[i] = epilogue.evaluate({
				acc,
				gidx: i
			}, globals);
		}
	}
	#getBuffer(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		return buffer.buffer;
	}
};
//#endregion
//#region src/tracing.ts
let traceEnabled = false;
const flushCallbacks = [];
/**
* Start collecting kernel traces.
*
* Traces appear in developer tools under the "Performance" tab, and they are
* useful for measuring fine-grained kernel execution time.
*/
function startTrace() {
	traceEnabled = true;
}
/**
* Stop collecting kernel traces.
*
* Traces appear in developer tools under the "Performance" tab, and they are
* useful for measuring fine-grained kernel execution time.
*/
function stopTrace() {
	traceEnabled = false;
	for (const cb of flushCallbacks) cb();
}
/** Check if tracing is currently enabled. */
function isTracing() {
	return traceEnabled;
}
/** Register a callback to flush pending trace data when tracing stops. */
function onFlushTrace(cb) {
	flushCallbacks.push(cb);
}
function humanSize(n) {
	if (n >= 1e9) return `${(n / 1e9).toPrecision(3)}B`;
	if (n >= 1e6) return `${(n / 1e6).toPrecision(3)}M`;
	if (n >= 1e3) return `${(n / 1e3).toPrecision(3)}K`;
	return `${n}`;
}
/** Build a trace label, properties, and color from a kernel or routine source. */
function traceSourceInfo(source) {
	const properties = [];
	let label;
	let color;
	if (source instanceof Kernel) {
		label = `Kernel[${humanSize(source.size)}]`;
		properties.push(["exp", `${source.exp}`]);
		properties.push(["size", `${source.size}`]);
		properties.push(["nargs", `${source.nargs}`]);
		if (!source.reduction) color = "primary";
		else {
			color = "secondary";
			properties.push(["reduction", `${source.reduction.op}:${source.reduction.size}`]);
		}
	} else {
		color = "tertiary";
		label = source.name;
		properties.push(["inputShapes", source.type.inputShapes.map((s) => `[${s}]`).join(", ")]);
		properties.push(["outputShapes", source.type.outputShapes.map((s) => `[${s}]`).join(", ")]);
		properties.push(["dtype", source.type.inputDtypes.join(", ")]);
	}
	return {
		label,
		color,
		properties
	};
}
/** Emit a trace entry as a `performance.measure` with devtools metadata. */
function emitTrace(track, info, start, end) {
	performance.measure(info.label, {
		detail: { devtools: {
			trackGroup: "JAX Profiler",
			track,
			color: info.color,
			properties: info.properties
		} },
		start,
		end
	});
}
//#endregion
//#region src/backend/wasm/allocator.ts
const ALLOCATION_ALIGNMENT = 64;
const WASM_PAGE_SIZE = 65536;
const MAX_MEMORY32_BYTES = 2 ** 32;
function alignTo(size, alignment) {
	return Math.ceil(size / alignment) * alignment;
}
/** Simple tensor memory allocator for WebAssembly linear memory. */
var WasmAllocator = class {
	#memory;
	#headPtr;
	#freeLists;
	#allocatedBuffers;
	constructor(memory) {
		this.#memory = memory;
		this.#headPtr = 64;
		this.#freeLists = /* @__PURE__ */ new Map();
		this.#allocatedBuffers = /* @__PURE__ */ new Map();
	}
	malloc(size) {
		if (size === 0) return 0;
		const sizeClass = this.#findSizeClass(size);
		const freeList = this.#freeLists.get(sizeClass);
		let ptr;
		if (freeList && freeList.length > 0) ptr = freeList.pop();
		else ptr = this.#bumpAlloc(sizeClass);
		this.#allocatedBuffers.set(ptr, sizeClass);
		return ptr;
	}
	free(ptr) {
		if (ptr === 0) return;
		const sizeClass = this.#allocatedBuffers.get(ptr);
		if (sizeClass === void 0) throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
		const freeList = this.#freeLists.get(sizeClass);
		if (freeList) freeList.push(ptr);
		else this.#freeLists.set(sizeClass, [ptr]);
		this.#allocatedBuffers.delete(ptr);
	}
	#bumpAlloc(size) {
		const ptr = this.#headPtr;
		const endPtr = ptr + alignTo(size, ALLOCATION_ALIGNMENT);
		if (endPtr > MAX_MEMORY32_BYTES) throw new RangeError("Allocation exceeds the 4 GiB memory32 limit");
		const currentBytes = this.#memory.buffer.byteLength;
		if (endPtr > currentBytes) {
			const requiredPages = Math.ceil(endPtr / WASM_PAGE_SIZE);
			const currentPages = currentBytes / WASM_PAGE_SIZE;
			this.#memory.grow(requiredPages - currentPages);
		}
		this.#headPtr = endPtr;
		return ptr;
	}
	#findSizeClass(size) {
		if (size <= 512) return alignTo(size, 64);
		if (size <= 2048) return alignTo(size, 512);
		if (size <= 65536) {
			let sizeClass = 4096;
			while (sizeClass < size) sizeClass *= 2;
			return sizeClass;
		}
		return alignTo(size, WASM_PAGE_SIZE);
	}
	getStats() {
		const freeListSizes = /* @__PURE__ */ new Map();
		for (const [sizeClass, freeList] of this.#freeLists) if (freeList.length > 0) freeListSizes.set(sizeClass, freeList.length);
		return {
			totalAllocated: this.#headPtr,
			freeListSizes
		};
	}
};
//#endregion
//#region src/backend/wasm/builtins.ts
/** Given a local `x`, evaluate `sum[i](a_i * x^i)` and push to stack. */
function _poly(cg, x, as) {
	if (as.length === 0) throw new Error("_poly needs at least one coefficient");
	cg.f32.const(as[as.length - 1]);
	for (let i = as.length - 2; i >= 0; i--) {
		cg.local.get(x);
		cg.f32.mul();
		if (as[i] !== 0) {
			cg.f32.const(as[i]);
			cg.f32.add();
		}
	}
}
/**
* Approximate e^x.
*
* Method: range-reduce x = k*ln2 + r with k = round(x/ln2), |r|<=~0.3466
*         then e^x = 2^k * P(r), where P is 5th-order poly (Taylor).
*/
function wasm_exp(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const k_f = cg.local.declare(cg.f32);
		const k = cg.local.declare(cg.i32);
		const r = cg.local.declare(cg.f32);
		const p = cg.local.declare(cg.f32);
		const scale = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.const(1 / Math.LN2);
		cg.f32.mul();
		cg.f32.nearest();
		cg.local.tee(k_f);
		cg.i32.trunc_sat_f32_s();
		cg.local.set(k);
		cg.local.get(k);
		cg.i32.const(127);
		cg.i32.gt_s();
		cg.if(cg.void);
		cg.f32.const(Infinity);
		cg.return();
		cg.end();
		cg.local.get(k);
		cg.i32.const(-126);
		cg.i32.lt_s();
		cg.if(cg.void);
		cg.f32.const(0);
		cg.return();
		cg.end();
		cg.local.get(0);
		cg.local.get(k_f);
		cg.f32.const(Math.LN2);
		cg.f32.mul();
		cg.f32.sub();
		cg.local.set(r);
		_poly(cg, r, [
			1,
			1,
			1 / 2,
			1 / 6,
			1 / 24,
			1 / 120,
			1 / 720
		]);
		cg.local.set(p);
		cg.local.get(k);
		cg.i32.const(127);
		cg.i32.add();
		cg.i32.const(23);
		cg.i32.shl();
		cg.f32.reinterpret_i32();
		cg.local.set(scale);
		cg.local.get(p);
		cg.local.get(scale);
		cg.f32.mul();
	});
}
/**
* Approximate ln(x), x > 0.
*
* Method: decompose x = m * 2^e with m in [1,2), e integer (via bit ops)
*         ln(x) = e*ln2 + ln(m);  use atanh-style series with t=(m-1)/(m+1)
*         ln(m) ≈ 2*(t + t^3/3 + t^5/5 + t^7/7)
*/
function wasm_log(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const bits = cg.local.declare(cg.i32);
		const e = cg.local.declare(cg.i32);
		const m = cg.local.declare(cg.f32);
		const t = cg.local.declare(cg.f32);
		const t2 = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.const(0);
		cg.f32.lt();
		cg.if(cg.void);
		cg.f32.const(NaN);
		cg.return();
		cg.end();
		cg.local.get(0);
		cg.i32.reinterpret_f32();
		cg.local.tee(bits);
		cg.i32.const(23);
		cg.i32.shr_u();
		cg.i32.const(255);
		cg.i32.and();
		cg.i32.const(127);
		cg.i32.sub();
		cg.local.set(e);
		cg.local.get(e);
		cg.i32.const(-127);
		cg.i32.eq();
		cg.if(cg.void);
		cg.f32.const(-Infinity);
		cg.return();
		cg.end();
		cg.local.get(e);
		cg.i32.const(128);
		cg.i32.eq();
		cg.if(cg.void);
		cg.local.get(0);
		cg.return();
		cg.end();
		cg.local.get(bits);
		cg.i32.const(8388607);
		cg.i32.and();
		cg.i32.const(1065353216);
		cg.i32.or();
		cg.f32.reinterpret_i32();
		cg.local.set(m);
		cg.local.get(m);
		cg.f32.const(1);
		cg.f32.sub();
		cg.local.get(m);
		cg.f32.const(1);
		cg.f32.add();
		cg.f32.div();
		cg.local.set(t);
		cg.local.get(t);
		cg.local.get(t);
		cg.f32.mul();
		cg.local.set(t2);
		_poly(cg, t2, [
			2,
			2 / 3,
			2 / 5,
			2 / 7
		]);
		cg.local.get(t);
		cg.f32.mul();
		cg.local.get(e);
		cg.f32.convert_i32_s();
		cg.f32.const(Math.LN2);
		cg.f32.mul();
		cg.f32.add();
	});
}
/**
* Common helper to approximate sin(x) and cos(x).
*
* Method: reduce to y in [-π, π], then quadrant via q = round(y/(π/2))
*         z = y - q*(π/2); use one of two polynomials on z:
*         sin(z) ≈ z + z^3*(-1/6) + z^5*(1/120) + z^7*(-1/5040)
*         cos(z) ≈ 1 + z^2*(-1/2) + z^4*(1/24) + z^6*(-1/720) + z^8*(1/40320)
*/
function _sincos(cg) {
	const y = cg.local.declare(cg.f32);
	const qf = cg.local.declare(cg.f32);
	const q = cg.local.declare(cg.i32);
	const z = cg.local.declare(cg.f32);
	const z2 = cg.local.declare(cg.f32);
	const sz = cg.local.declare(cg.f32);
	const cz = cg.local.declare(cg.f32);
	cg.local.get(0);
	cg.local.get(0);
	cg.f32.const(1 / (2 * Math.PI));
	cg.f32.mul();
	cg.f32.nearest();
	cg.local.tee(qf);
	cg.f32.const(2 * Math.PI);
	cg.f32.mul();
	cg.f32.sub();
	cg.local.set(y);
	cg.local.get(y);
	cg.f32.const(2 / Math.PI);
	cg.f32.mul();
	cg.f32.nearest();
	cg.local.tee(qf);
	cg.i32.trunc_sat_f32_s();
	cg.local.set(q);
	cg.local.get(y);
	cg.local.get(qf);
	cg.f32.const(Math.PI / 2);
	cg.f32.mul();
	cg.f32.sub();
	cg.local.tee(z);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(z2);
	_poly(cg, z2, [
		1,
		-1 / 6,
		1 / 120,
		-1 / 5040
	]);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(sz);
	_poly(cg, z2, [
		1,
		-1 / 2,
		1 / 24,
		-1 / 720,
		1 / 40320
	]);
	cg.local.set(cz);
	return {
		q,
		sz,
		cz
	};
}
/**
* Approximate sin(x).
*
* Quadrant mapping: k=q mod 4: 0: +sz, 1: +cz, 2: -sz, 3: -cz
*/
function wasm_sin(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const { q, sz, cz } = _sincos(cg);
		const mag = cg.local.declare(cg.f32);
		cg.local.get(cz);
		cg.local.get(sz);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.and();
		cg.select();
		cg.local.tee(mag);
		cg.f32.neg();
		cg.local.get(mag);
		cg.local.get(q);
		cg.i32.const(2);
		cg.i32.and();
		cg.select();
	});
}
/**
* Approximate cos(x).
*
* Quadrant mapping: k=q mod 4: 0: +cz, 1: -sz, 2: -cz, 3: +sz
*/
function wasm_cos(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const { q, sz, cz } = _sincos(cg);
		const mag = cg.local.declare(cg.f32);
		cg.local.get(sz);
		cg.local.get(cz);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.and();
		cg.select();
		cg.local.tee(mag);
		cg.f32.neg();
		cg.local.get(mag);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.add();
		cg.i32.const(2);
		cg.i32.and();
		cg.select();
	});
}
/** Helper function for approximating arctan(x).  */
function _atan(cg) {
	const x = cg.local.declare(cg.f32);
	const abs_x = cg.local.declare(cg.f32);
	const z = cg.local.declare(cg.f32);
	const z2 = cg.local.declare(cg.f32);
	const p = cg.local.declare(cg.f32);
	cg.local.set(x);
	cg.local.get(x);
	cg.f32.abs();
	cg.local.set(abs_x);
	cg.f32.const(1);
	cg.local.get(abs_x);
	cg.f32.div();
	cg.local.get(abs_x);
	cg.local.get(abs_x);
	cg.f32.const(1);
	cg.f32.ge();
	cg.select();
	cg.local.set(z);
	cg.local.get(z);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(z2);
	_poly(cg, z2, [
		.999998614341,
		.661705427875,
		.0415796528637
	]);
	_poly(cg, z2, [
		1,
		.994987933645,
		.173698870181
	]);
	cg.f32.div();
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(p);
	cg.f32.const(Math.PI / 2);
	cg.local.get(p);
	cg.f32.sub();
	cg.local.get(p);
	cg.local.get(abs_x);
	cg.f32.const(1);
	cg.f32.ge();
	cg.select();
	cg.local.get(x);
	cg.f32.copysign();
}
/**
* Approximate atan(x).
*
* Method: if |x| < 1, use rational approximation: atan(x) ≈ x * P(x^2) / Q(x^2)
*         where P(u) = A0 + A1*u + A2*u^2 (degree 2)
*               Q(u) = 1 + B1*u + B2*u^2 (degree 2)
*         if |x| >= 1, use: atan(x) = sign(x)*π/2 - atan(1/x)
*         (fitted coefficients, max error ~5e-7 on [0,1])
*/
function wasm_atan(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.local.get(0);
		_atan(cg);
	});
}
/**
* Approximate asin(x).
*
* Method: asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
*/
function wasm_asin(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.local.get(0);
		cg.f32.const(1);
		cg.local.get(0);
		cg.local.get(0);
		cg.f32.mul();
		cg.f32.sub();
		cg.f32.sqrt();
		cg.f32.const(1);
		cg.f32.add();
		cg.f32.div();
		_atan(cg);
		cg.f32.const(2);
		cg.f32.mul();
	});
}
/**
* Helper function for erf/erfc approximation.
*
* See `_erfapprox` in alu.ts for details on the algorithm used.
*/
function _erfapprox(cg, exp_func) {
	const x = cg.local.declare(cg.f32);
	const t = cg.local.declare(cg.f32);
	cg.local.set(x);
	const p = .3275911;
	const a1 = .254829592;
	const a2 = -.284496736;
	const a3 = 1.421413741;
	const a4 = -1.453152027;
	const a5 = 1.061405429;
	cg.f32.const(1);
	cg.f32.const(1);
	cg.f32.const(p);
	cg.local.get(x);
	cg.f32.mul();
	cg.f32.add();
	cg.f32.div();
	cg.local.set(t);
	_poly(cg, t, [
		0,
		a1,
		a2,
		a3,
		a4,
		a5
	]);
	cg.local.get(x);
	cg.f32.neg();
	cg.local.get(x);
	cg.f32.mul();
	cg.call(exp_func);
	cg.f32.mul();
}
/** Approximate erf(x) (error function). */
function wasm_erf(cg, exp) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.f32.const(1);
		cg.local.get(0);
		cg.f32.abs();
		_erfapprox(cg, exp);
		cg.f32.sub();
		cg.local.get(0);
		cg.f32.copysign();
	});
}
/** Approximate erfc(x) (complementary error function). */
function wasm_erfc(cg, exp) {
	return cg.function([cg.f32], [cg.f32], () => {
		const e = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.abs();
		_erfapprox(cg, exp);
		cg.local.set(e);
		cg.f32.const(2);
		cg.local.get(e);
		cg.f32.sub();
		cg.local.get(e);
		cg.local.get(0);
		cg.f32.const(0);
		cg.f32.lt();
		cg.select();
	});
}
/**
* Threefry2x32 pseudorandom number generator.
*
* Takes two 32-bit keys and two 32-bit counters as input,
* returns two 32-bit pseudorandom values.
*/
function wasm_threefry2x32(cg) {
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [cg.i32, cg.i32], () => {
		const ks0 = cg.local.declare(cg.i32);
		const ks1 = cg.local.declare(cg.i32);
		const ks2 = cg.local.declare(cg.i32);
		const x0 = cg.local.declare(cg.i32);
		const x1 = cg.local.declare(cg.i32);
		const mix = (rot) => {
			cg.local.get(x0);
			cg.local.get(x1);
			cg.i32.add();
			cg.local.set(x0);
			cg.local.get(x1);
			cg.i32.const(rot);
			cg.i32.rotl();
			cg.local.get(x0);
			cg.i32.xor();
			cg.local.set(x1);
		};
		const keySchedule = (k0, k1, round) => {
			cg.local.get(x0);
			cg.local.get(k0);
			cg.i32.add();
			cg.local.set(x0);
			cg.local.get(x1);
			cg.local.get(k1);
			cg.i32.add();
			cg.i32.const(round);
			cg.i32.add();
			cg.local.set(x1);
		};
		cg.local.get(0);
		cg.local.set(ks0);
		cg.local.get(1);
		cg.local.set(ks1);
		cg.local.get(0);
		cg.local.get(1);
		cg.i32.xor();
		cg.i32.const(466688986);
		cg.i32.xor();
		cg.local.set(ks2);
		cg.local.get(2);
		cg.local.get(ks0);
		cg.i32.add();
		cg.local.set(x0);
		cg.local.get(3);
		cg.local.get(ks1);
		cg.i32.add();
		cg.local.set(x1);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks1, ks2, 1);
		mix(17), mix(29), mix(16), mix(24);
		keySchedule(ks2, ks0, 2);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks0, ks1, 3);
		mix(17), mix(29), mix(16), mix(24);
		keySchedule(ks1, ks2, 4);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks2, ks0, 5);
		cg.local.get(x0);
		cg.local.get(x1);
	});
}
//#endregion
//#region src/backend/wasm/featureProbe.ts
const featureProbes = { "relaxed-madd": "0061736d0100000001080160037b7b7b017b030201000a0d010b00200020012002fd85020b" };
const featureSupportCache = /* @__PURE__ */ new Map();
/** Detects whether this environment supports a probed Wasm CPU feature. */
function hasWasmFeature(feature) {
	const cached = featureSupportCache.get(feature);
	if (cached !== void 0) return cached;
	const testHex = featureProbes[feature];
	let supported = false;
	try {
		supported = typeof WebAssembly !== "undefined" && WebAssembly.validate(decodeHex(testHex));
	} catch {
		supported = false;
	}
	featureSupportCache.set(feature, supported);
	return supported;
}
function decodeHex(hex) {
	const bytes = new Uint8Array(/* @__PURE__ */ new ArrayBuffer(hex.length / 2));
	for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
	return bytes;
}
//#endregion
//#region src/backend/wasm/parallel.ts
/** Check if SharedArrayBuffer is available. */
function hasSharedArrayBuffer() {
	return typeof SharedArrayBuffer !== "undefined" && typeof Worker !== "undefined";
}
const MIN_ELEMS_PER_THREAD = 512;
const WORKER_SOURCE = `
${cpuRoutineJSForWorkers()}

let memory = null;

const kernelLru = new Map();
function kernelFunc(kernelHash, module) {
  let func = kernelLru.get(kernelHash);
  if (func !== undefined) {
    kernelLru.delete(kernelHash);
    kernelLru.set(kernelHash, func);
    return func;
  }
  const instance = new WebAssembly.Instance(module, { env: { memory } });
  func = instance.exports.kernel;
  kernelLru.set(kernelHash, func);
  if (kernelLru.size > 256) {
    kernelLru.delete(kernelLru.keys().next().value);
  }
  return func;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    postMessage({ type: "ready" });
    return;
  }
  try {
    if (msg.type === "kernel") {
      const { kernelHash, module, ptrs, begin, end } = msg;
      kernelFunc(kernelHash, module)(...ptrs, begin, end);
    } else if (msg.type === "routine") {
      const inputs = msg.inputs.map(({ ptr, size }) =>
        new Uint8Array(memory.buffer, ptr, size)
      );
      const outputs = msg.outputs.map(({ ptr, size }) =>
        new Uint8Array(memory.buffer, ptr, size)
      );
      __minify_safe_runCpuRoutine(msg.routine, inputs, outputs);
    } else {
      throw new Error("Unknown wasm worker message: " + msg.type);
    }
    postMessage({ type: "done", ok: true });
  } catch (err) {
    postMessage({ type: "done", ok: false, error: String(err) });
  }
};
`;
/** Pool of Web Workers for parallel WASM kernel dispatch. */
var WasmWorkerPool = class {
	#memory;
	#numWorkers;
	#workers = [];
	#ready = Promise.resolve();
	/** Serializes dispatches so concurrent read() calls don't clobber onmessage. */
	#queue = Promise.resolve();
	#epoch = 0n;
	#epochEnd = 0n;
	#hooks = /* @__PURE__ */ new Map();
	constructor(memory, numWorkers) {
		if (numWorkers <= 0) throw new Error("numWorkers must be positive");
		this.#memory = memory;
		this.#numWorkers = numWorkers;
	}
	get epoch() {
		return this.#epoch;
	}
	waitForEpoch(target) {
		if (target <= this.#epoch) return Promise.resolve();
		return new Promise((resolve) => {
			if (target <= this.#epoch) return resolve();
			const hooks = this.#hooks.get(target);
			if (hooks) hooks.push(resolve);
			else this.#hooks.set(target, [resolve]);
		});
	}
	#ensureInit() {
		if (this.#workers.length > 0) return;
		const blob = new Blob([WORKER_SOURCE], { type: "application/javascript" });
		const url = URL.createObjectURL(blob);
		this.#workers = [];
		const readyPromises = [];
		for (let i = 0; i < this.#numWorkers; i++) {
			const worker = new Worker(url, { type: "module" });
			this.#workers.push(worker);
			readyPromises.push(new Promise((resolve, reject) => {
				worker.onmessage = () => resolve();
				worker.onerror = (e) => reject(new Error(e.message || "Worker failed to load"));
			}));
			worker.postMessage({
				type: "init",
				memory: this.#memory
			});
		}
		this.#ready = Promise.all(readyPromises).then(() => {
			URL.revokeObjectURL(url);
		});
		this.#queue = this.#ready;
	}
	/**
	* Dispatch a kernel across multiple workers.
	*
	* Returns an epoch that can be used to wait for the ongoing work to complete,
	* which is guaranteed to be monotonically increasing.
	*/
	dispatch(kernelHash, module, ptrs, size, chunkAlignment = 16, minWorkPerWorker = MIN_ELEMS_PER_THREAD) {
		return this.#enqueue(() => this.#dispatchNow(kernelHash, module, ptrs, size, chunkAlignment, minWorkPerWorker));
	}
	/**
	* Dispatch a CPU fallback routine on a worker.
	*
	* This uses the same serial queue as generated wasm kernels so routine reads
	* observe prior async writes and later kernels observe routine outputs.
	*/
	dispatchRoutine(routine, inputs, outputs) {
		return this.#enqueue(() => this.#dispatchRoutineNow(routine, inputs, outputs));
	}
	#enqueue(job) {
		this.#ensureInit();
		this.#epochEnd++;
		const result = this.#queue.then(job);
		this.#queue = result.then(() => {}, () => {}).then(() => {
			this.#epoch++;
			const hooks = this.#hooks.get(this.#epoch);
			if (hooks) {
				for (const hook of hooks) hook();
				this.#hooks.delete(this.#epoch);
			}
		});
		return this.#epochEnd;
	}
	async #dispatchNow(kernelHash, module, ptrs, size, chunkAlignment, minWorkPerWorker) {
		if (size === 0) return;
		const n = Math.min(this.#workers.length, Math.ceil(size / minWorkPerWorker));
		const chunkSize = Math.ceil(size / n / chunkAlignment) * chunkAlignment;
		const promises = [];
		for (let i = 0; i < n; i++) {
			const begin = i * chunkSize;
			const end = Math.min(begin + chunkSize, size);
			if (begin >= size) break;
			const worker = this.#workers[i];
			promises.push(new Promise((resolve, reject) => {
				worker.onmessage = (e) => {
					if (e.data.ok) resolve();
					else reject(/* @__PURE__ */ new Error(`Worker error: ${e.data.error}`));
				};
				worker.postMessage({
					type: "kernel",
					kernelHash,
					module,
					ptrs,
					begin,
					end
				});
			}));
		}
		await Promise.all(promises);
	}
	async #dispatchRoutineNow(routine, inputs, outputs) {
		const worker = this.#workers[0];
		await new Promise((resolve, reject) => {
			worker.onmessage = (e) => {
				if (e.data.ok) resolve();
				else reject(/* @__PURE__ */ new Error(`Worker error: ${e.data.error}`));
			};
			worker.postMessage({
				type: "routine",
				routine,
				inputs,
				outputs
			});
		});
	}
};
/** Try to create a worker pool. Returns null if workers are unavailable. */
function createWorkerPool(memory) {
	if (!hasSharedArrayBuffer()) return null;
	try {
		return new WasmWorkerPool(memory, Math.max(1, typeof navigator !== "undefined" && navigator.hardwareConcurrency || 4));
	} catch {
		return null;
	}
}
const TILED_SIMD_ROWS = 128;
const TILED_SIMD_COLUMNS = 128;
const TILED_SIMD_K = 64;
const TILED_SIMD_MICRO_ROWS = 4;
const TILED_SIMD_MICRO_VECTORS = 4;
const K_SIMD_MICRO_ROWS = 4;
const K_SIMD_MICRO_COLS = 4;
const K_SIMD_MAX_UNROLL = 4;
function isSymbol(exp, name) {
	return exp.op === "Variable" && exp.arg === name || exp.op === "Special" && exp.arg[0] === name;
}
function referencesSymbol(exp, name) {
	return isSymbol(exp, name) || exp.src.some((src) => referencesSymbol(src, name));
}
function referencesGidx(exp) {
	return referencesSymbol(exp, "gidx");
}
function hasFragmentRisk(tileSize, N) {
	return isFinite(tileSize) && tileSize > N && tileSize % N !== 0;
}
function constInt(exp) {
	if (exp.op !== "Const") return null;
	const value = exp.arg;
	return Number.isInteger(value) ? value : null;
}
function coefficientOfSymbol(exp, name) {
	if (!referencesSymbol(exp, name)) return 0;
	if (isSymbol(exp, name)) return 1;
	if (exp.op === "Add" || exp.op === "Sub") {
		const a = coefficientOfSymbol(exp.src[0], name);
		const b = coefficientOfSymbol(exp.src[1], name);
		if (a === null || b === null) return null;
		return exp.op === "Add" ? a + b : a - b;
	}
	if (exp.op === "Mul") {
		const lhs = constInt(exp.src[0]);
		if (lhs !== null) {
			const rhsCoeff = coefficientOfSymbol(exp.src[1], name);
			return rhsCoeff === null ? null : lhs * rhsCoeff;
		}
		const rhs = constInt(exp.src[1]);
		if (rhs !== null) {
			const lhsCoeff = coefficientOfSymbol(exp.src[0], name);
			return lhsCoeff === null ? null : rhs * lhsCoeff;
		}
	}
	return null;
}
function rewriteSymbol(exp, name, rewrite) {
	return exp.rewrite((node) => isSymbol(node, name) ? rewrite(node) : void 0).simplify();
}
function repeatsAcrossGidxTile(exp, tileSize) {
	if (!isFinite(tileSize)) return false;
	return rewriteSymbol(exp, "gidx", (node) => AluExp.add(node, AluExp.i32(tileSize))).getHash() === exp.getHash();
}
function divisorAtMost(value, limit) {
	for (let i = Math.min(value, limit); i > 1; i--) if (value % i === 0) return i;
	return 1;
}
function commonTileSize(kernelSize, strideMap, minTileSize, unconstrainedTileSize = null) {
	const tileSizes = [];
	for (const stride of strideMap.values()) if (stride.kind !== "gather" && isFinite(stride.tileSize)) tileSizes.push(stride.tileSize);
	if (tileSizes.length === 0) return unconstrainedTileSize;
	const tileSize = Math.min(...tileSizes);
	if (tileSize < minTileSize || kernelSize % tileSize !== 0 || tileSizes.some((size) => size % tileSize !== 0)) return null;
	return tileSize;
}
function tiledRows(kernelSize, tileSize) {
	const rowCount = Math.floor(kernelSize / tileSize);
	return findPow2(0, Math.max(1, Math.min(rowCount / 2, TILED_SIMD_ROWS)));
}
function tiledColumns(tileSize, laneWidth) {
	const columns = tileSize / laneWidth;
	return findPow2(0, Math.max(1, Math.min(columns / 2, Math.floor(TILED_SIMD_COLUMNS / laneWidth))));
}
function microTile(tile, axis, limit) {
	return divisorAtMost(gcd(tile, axis), limit);
}
function microTileWithTail(tile, limit) {
	return Math.max(1, Math.min(tile, limit));
}
function periodicStride(exp, kind) {
	if (exp.src[1].op !== "Const") return null;
	const N = exp.src[1].arg;
	const inner = analyzeStride(exp.src[0]);
	if (inner.kind === "broadcast") return inner;
	if (inner.kind !== "contiguous" || hasFragmentRisk(inner.tileSize, N)) return { kind: "gather" };
	return {
		kind,
		tileSize: Math.min(inner.tileSize, N)
	};
}
function addStrides(lhs, rhs) {
	if (lhs.kind === "gather" || rhs.kind === "gather") return { kind: "gather" };
	const tileSize = Math.min(lhs.tileSize, rhs.tileSize);
	if (lhs.kind === "broadcast") return {
		kind: rhs.kind,
		tileSize
	};
	if (rhs.kind === "broadcast") return {
		kind: lhs.kind,
		tileSize
	};
	return { kind: "gather" };
}
function analyzeStride(exp) {
	if (!referencesGidx(exp)) return {
		kind: "broadcast",
		tileSize: Infinity
	};
	if (exp.op === "Special" && exp.arg[0] === "gidx") return {
		kind: "contiguous",
		tileSize: Infinity
	};
	if (exp.op === "Idiv" || exp.op === "Mod") {
		const stride = periodicStride(exp, exp.op === "Idiv" ? "broadcast" : "contiguous");
		if (stride) return stride;
	}
	if (exp.op === "Mul") {
		for (let i = 0; i < 2; i++) if (exp.src[i].op === "Const") {
			const inner = analyzeStride(exp.src[1 - i]);
			if (inner.kind === "broadcast") return inner;
			return { kind: "gather" };
		}
	}
	if (exp.op === "Add") return addStrides(analyzeStride(exp.src[0]), analyzeStride(exp.src[1]));
	return { kind: "gather" };
}
function simdStrideResult(globalIndex) {
	const index = globalIndex.src[0];
	const result = analyzeStride(index);
	const [_, len] = globalIndex.arg;
	if (result.kind !== "gather" && (result.tileSize < 4 || isFinite(result.tileSize) && result.tileSize % 4 !== 0)) return { kind: "gather" };
	if (result.kind === "contiguous" && (index.min < 0 || index.max >= len)) return { kind: "gather" };
	return result;
}
function collectSimdStrides(exp) {
	return new Map((exp?.collect((node) => node.op === "GlobalIndex") ?? []).map((gi) => [gi, simdStrideResult(gi)]));
}
function reductionPointerCandidates(exp, strideMap) {
	const candidates = [];
	for (const gi of exp.collect((node) => node.op === "GlobalIndex")) {
		const stride = strideMap?.get(gi) ?? {
			kind: "broadcast",
			tileSize: Infinity
		};
		if (strideMap && stride.kind === "gather") continue;
		const [gid, len] = gi.arg;
		const index = gi.src[0];
		const strideElems = coefficientOfSymbol(index, "ridx");
		if (strideElems === null || !Number.isInteger(strideElems)) continue;
		if (index.min < 0 || index.max >= len) continue;
		candidates.push({
			exp: gi,
			gid,
			dtype: gi.dtype,
			stride,
			baseIndex: rewriteSymbol(index, "ridx", () => AluExp.i32(0)),
			strideBytes: strideElems * byteWidth(gi.dtype)
		});
	}
	return candidates;
}
function reductionTilePlan(kernel, strideMap, canStorePartial = true) {
	if (!kernel.reduction) return null;
	const tileSize = commonTileSize(kernel.size, strideMap, 4, 4);
	if (tileSize === null) return null;
	const tileRows = tiledRows(kernel.size, tileSize);
	const tileVectors = tiledColumns(tileSize, 4);
	return {
		tileSize,
		tileRows,
		tileVectors,
		tileK: canStorePartial ? divisorAtMost(kernel.reduction.size, TILED_SIMD_K) : kernel.reduction.size,
		microRows: microTile(tileRows, Math.floor(kernel.size / tileSize), TILED_SIMD_MICRO_ROWS),
		microVectors: microTileWithTail(tileVectors, TILED_SIMD_MICRO_VECTORS)
	};
}
function reductionKTilePlan(kernel, strideMap) {
	if (!kernel.reduction) return null;
	if (kernel.reduction.size % 4 !== 0) return null;
	const tileSize = commonTileSize(kernel.size, strideMap, 1, 1);
	if (tileSize === null) return null;
	const kUnroll = divisorAtMost(kernel.reduction.size / 4, K_SIMD_MAX_UNROLL);
	const tileRows = tiledRows(kernel.size, tileSize);
	const tileCols = tiledColumns(tileSize, 1);
	return {
		tileSize,
		tileRows,
		tileCols,
		microRows: microTile(tileRows, Math.floor(kernel.size / tileSize), K_SIMD_MICRO_ROWS),
		microCols: microTile(tileCols, tileSize, K_SIMD_MICRO_COLS),
		kUnroll
	};
}
function pointerShareKey(candidate, row, vector, groupIndex) {
	const hash = candidate.exp.getHash().toString();
	const { stride } = candidate;
	if (stride.kind === "broadcast") return isFinite(stride.tileSize) ? `${hash}:row${row}` : `${hash}:all`;
	if (stride.kind === "contiguous" && isFinite(stride.tileSize)) return repeatsAcrossGidxTile(candidate.baseIndex, stride.tileSize) ? `${hash}:vec${vector}` : `${hash}:row${row}:vec${vector}`;
	return `${hash}:g${groupIndex}`;
}
function kReductionPointerShareKey(candidate, outputStride, tileSize, row, col, groupIndex) {
	const hash = candidate.exp.getHash().toString();
	if (outputStride.kind === "broadcast" && isFinite(outputStride.tileSize)) return `${hash}:row${row}`;
	if (repeatsAcrossGidxTile(candidate.baseIndex, tileSize)) return `${hash}:col${col}`;
	return `${hash}:g${groupIndex}`;
}
//#endregion
//#region src/backend/wasm/translation.ts
function translateExp(cg, funcs, exp, ctx, pointerMap = /* @__PURE__ */ new Map()) {
	const references = /* @__PURE__ */ new Map();
	const seen = /* @__PURE__ */ new Set();
	const countReferences = (exp) => {
		references.set(exp, (references.get(exp) ?? 0) + 1);
		if (!seen.has(exp)) {
			seen.add(exp);
			for (const src of exp.src) countReferences(src);
		}
	};
	const expContext = /* @__PURE__ */ new Map();
	const gen = (exp) => {
		if (expContext.has(exp)) return cg.local.get(expContext.get(exp));
		const { op, src, dtype, arg } = exp;
		if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
			gen(src[0]), gen(src[1]);
			if (op === "Add") if (dtype === "bool") cg.i32.or();
			else dty(cg, op, dtype).add();
			else if (op === "Sub") dty(cg, op, dtype).sub();
			else if (op === "Mul") if (dtype === "bool") cg.i32.and();
			else dty(cg, op, dtype).mul();
			else if (op === "Idiv") if (isFloatDtype(dtype)) {
				dtyF(cg, op, dtype).div();
				dtyF(cg, op, dtype).trunc();
			} else if (dtype === "uint32") cg.i32.div_u();
			else if (dtype === "int32") cg.i32.div_s();
			else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === "Mod") if (isFloatDtype(dtype)) {
				const dt = dtyF(cg, op, dtype);
				const a = cg.local.declare(dt);
				const b = cg.local.declare(dt);
				cg.local.set(b);
				cg.local.tee(a);
				cg.local.get(a);
				cg.local.get(b);
				dt.div();
				dt.trunc();
				cg.local.get(b);
				dt.mul();
				dt.sub();
			} else if (dtype === "uint32") cg.i32.rem_u();
			else if (dtype === "int32") cg.i32.rem_s();
			else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === "Min" || op === "Max") if (isFloatDtype(dtype)) if (op === "Min") dtyF(cg, op, dtype).min();
			else dtyF(cg, op, dtype).max();
			else if (dtype === "int32" || dtype === "uint32" || dtype === "bool") {
				const a = cg.local.declare(cg.i32);
				const b = cg.local.declare(cg.i32);
				cg.local.set(b);
				cg.local.tee(a);
				cg.local.get(b);
				cg.local.get(a);
				cg.local.get(b);
				if (dtype === "int32") if (op === "Min") cg.i32.lt_s();
				else cg.i32.gt_s();
				else if (op === "Min") cg.i32.lt_u();
				else cg.i32.gt_u();
				cg.select();
			} else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === "BitCombine") if (arg === "and") cg.i32.and();
			else if (arg === "or") cg.i32.or();
			else cg.i32.xor();
			else if (op === "BitShift") if (arg === "shl") cg.i32.shl();
			else cg.i32.shr_u();
			else if (op === "Cmplt") {
				const srcDtype = src[0].dtype;
				if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
				else if (srcDtype === "int32") cg.i32.lt_s();
				else if (srcDtype === "uint32") cg.i32.lt_u();
				else throw new UnsupportedOpError(op, dtype, "wasm");
			} else if (op === "Cmpne") dty(cg, op, src[0].dtype).ne();
			else throw new UnsupportedOpError(op, dtype, "wasm");
		} else if (AluGroup.Unary.has(op)) {
			const callFuncF32 = (func) => {
				if (dtype !== "float32") if (dtype === "float64") cg.f32.demote_f64();
				else throw new UnsupportedOpError(op, dtype, "wasm");
				cg.call(func);
				if (dtype === "float64") cg.f64.promote_f32();
			};
			if (op === "BitCount") gen(src[0]), cg.i32.popcnt();
			else if (op === "Sin") gen(src[0]), callFuncF32(funcs.sin);
			else if (op === "Cos") gen(src[0]), callFuncF32(funcs.cos);
			else if (op === "Asin") gen(src[0]), callFuncF32(funcs.asin);
			else if (op === "Atan") gen(src[0]), callFuncF32(funcs.atan);
			else if (op === "Exp") gen(src[0]), callFuncF32(funcs.exp);
			else if (op === "Log") gen(src[0]), callFuncF32(funcs.log);
			else if (op === "Erf") gen(src[0]), callFuncF32(funcs.erf);
			else if (op === "Erfc") gen(src[0]), callFuncF32(funcs.erfc);
			else if (op === "Sqrt") gen(src[0]), dtyF(cg, op, dtype).sqrt();
			else if (op === "Reciprocal") {
				const dt = dtyF(cg, op, dtype);
				dt.const(1), gen(src[0]), dt.div();
			} else if (op === "Floor") gen(src[0]), dtyF(cg, op, dtype).floor();
			else if (op === "Ceil") gen(src[0]), dtyF(cg, op, dtype).ceil();
			else if (op === "Cast") {
				gen(src[0]);
				const dtype0 = src[0].dtype;
				const i32repr = dtype0 === "int32" || dtype0 === "uint32" || dtype0 === "bool";
				if (dtype === "int32") if (dtype0 === "float32") cg.i32.trunc_sat_f32_s();
				else if (dtype0 === "float64") cg.i32.trunc_sat_f64_s();
				else if (i32repr);
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === "uint32") if (dtype0 === "float32") cg.i32.trunc_sat_f32_u();
				else if (dtype0 === "float64") cg.i32.trunc_sat_f64_u();
				else if (i32repr);
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === "float32") if (dtype0 === "float32");
				else if (dtype0 === "float64") cg.f32.demote_f64();
				else if (dtype0 === "int32" || dtype0 === "bool") cg.f32.convert_i32_s();
				else if (dtype0 === "uint32") cg.f32.convert_i32_u();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === "float64") if (dtype0 === "float32") cg.f64.promote_f32();
				else if (dtype0 === "float64");
				else if (dtype0 === "int32" || dtype0 === "bool") cg.f64.convert_i32_s();
				else if (dtype0 === "uint32") cg.f64.convert_i32_u();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === "bool") if (dtype0 === "bool");
				else if (i32repr) cg.i32.const(0), cg.i32.ne();
				else if (dtype0 === "float32") cg.f32.const(0), cg.f32.ne();
				else if (dtype0 === "float64") cg.f64.const(0), cg.f64.ne();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else throw new UnsupportedOpError(op, dtype, "wasm");
			} else if (op === "Bitcast") {
				gen(src[0]);
				const dtype0 = src[0].dtype;
				if (dtype !== dtype0) {
					const i32repr = dtype0 === "int32" || dtype0 === "uint32";
					if (dtype === "int32" || dtype === "uint32") if (dtype0 === "float32") cg.i32.reinterpret_f32();
					else if (i32repr);
					else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
					else if (dtype === "float32") if (i32repr) cg.f32.reinterpret_i32();
					else if (dtype0 === "float32");
					else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
					else throw new UnsupportedOpError(op, dtype, "wasm");
				}
			} else throw new UnsupportedOpError(op, dtype, "wasm");
		} else if (op === "Where") {
			gen(src[1]);
			gen(src[2]);
			gen(src[0]);
			cg.select();
		} else if (op === "Threefry2x32") {
			for (let i = 0; i < 4; i++) gen(src[i]);
			cg.call(funcs.threefry2x32);
			if (arg === "xor") cg.i32.xor();
			else if (arg === 0) cg.drop();
			else if (arg === 1) {
				const local = cg.local.declare(cg.i32);
				cg.local.set(local);
				cg.drop();
				cg.local.get(local);
			} else throw new UnsupportedOpError(op, dtype, "wasm", arg);
		} else if (op === "Const") return dty(cg, op, dtype).const(arg);
		else if (op === "Special") return cg.local.get(ctx[arg[0]]);
		else if (op === "Variable") return cg.local.get(ctx[arg]);
		else if (op === "GlobalIndex") {
			const [gid, len] = arg;
			const pointer = pointerMap.get(exp);
			if (pointer) cg.local.get(pointer.ptr);
			else {
				gen(src[0]);
				const local = cg.local.declare(cg.i32);
				cg.local.tee(local);
				cg.i32.const(0);
				cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
				cg.select();
				cg.i32.const(byteWidth(dtype));
				cg.i32.mul();
				cg.local.get(gid);
				cg.i32.add();
			}
			dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
		} else throw new UnsupportedOpError(op, dtype, "wasm");
		if ((references.get(exp) ?? 0) > 1) {
			const local = cg.local.declare(dty(cg, op, dtype));
			cg.local.tee(local);
			expContext.set(exp, local);
		}
	};
	countReferences(exp);
	gen(exp);
}
/**
* SIMD version of translateExp. Emits one v128 value for `exp`, interpreting
* the current `gidx` local as the first lane and the following lanes as
* `gidx + 1`, `gidx + 2`, etc.
*
* GlobalIndex loads are the only places where per-lane address behavior
* matters:
* - `strideMap` classifies each GlobalIndex as contiguous, broadcast, or
*   gather. Contiguous loads become one v128.load, broadcast loads become a
*   scalar load plus splat, and gather loads fall back to four scalar loads.
* - `pointerMap` optionally supplies precomputed reduction pointers for
*   GlobalIndex nodes whose address can be advanced by the surrounding
*   reduction loop. This avoids re-emitting scalar index math in the hot path.
* - `pointerValueCache` lets pointer plans with the same `valueKey` share a
*   single loaded vector within one emitted reduction step.
*/
function translateExpSimd(cg, funcs, exp, ctx, strideMap, pointerMap = /* @__PURE__ */ new Map(), pointerValueCache = /* @__PURE__ */ new Map()) {
	const references = /* @__PURE__ */ new Map();
	const seen = /* @__PURE__ */ new Set();
	const countReferences = (exp) => {
		references.set(exp, (references.get(exp) ?? 0) + 1);
		if (!seen.has(exp)) {
			seen.add(exp);
			for (const src of exp.src) countReferences(src);
		}
	};
	const expContext = /* @__PURE__ */ new Map();
	const gen = (exp) => {
		if (expContext.has(exp)) return cg.local.get(expContext.get(exp));
		const { op, src, arg, dtype } = exp;
		const isInt = dtype === "int32" || dtype === "uint32" || dtype === "bool";
		const isSigned = dtype === "int32";
		if (op === "Add") {
			gen(src[0]), gen(src[1]);
			if (dtype === "bool") cg.v128.or();
			else if (isInt) cg.i32x4.add();
			else cg.f32x4.add();
		} else if (op === "Sub") {
			gen(src[0]), gen(src[1]);
			if (isInt) cg.i32x4.sub();
			else cg.f32x4.sub();
		} else if (op === "Mul") {
			gen(src[0]), gen(src[1]);
			if (dtype === "bool") cg.v128.and();
			else if (isInt) cg.i32x4.mul();
			else cg.f32x4.mul();
		} else if (op === "Min") {
			gen(src[0]), gen(src[1]);
			if (isInt) if (isSigned) cg.i32x4.min_s();
			else cg.i32x4.min_u();
			else cg.f32x4.min();
		} else if (op === "Max") {
			gen(src[0]), gen(src[1]);
			if (isInt) if (isSigned) cg.i32x4.max_s();
			else cg.i32x4.max_u();
			else cg.f32x4.max();
		} else if (op === "Sqrt") {
			gen(src[0]);
			cg.f32x4.sqrt();
		} else if (op === "Floor") {
			gen(src[0]);
			cg.f32x4.floor();
		} else if (op === "Ceil") {
			gen(src[0]);
			cg.f32x4.ceil();
		} else if (op === "Const") if (isInt) {
			cg.i32.const(arg);
			cg.i32x4.splat();
		} else {
			cg.f32.const(arg);
			cg.f32x4.splat();
		}
		else if (op === "Cast") {
			gen(src[0]);
			const dtype0 = src[0].dtype;
			const src0IsInt = dtype0 === "int32" || dtype0 === "uint32" || dtype0 === "bool";
			if (isInt && !src0IsInt) if (isSigned) cg.i32x4.trunc_sat_f32x4_s();
			else cg.i32x4.trunc_sat_f32x4_u();
			else if (!isInt && src0IsInt) if (dtype0 === "int32" || dtype0 === "bool") cg.f32x4.convert_i32x4_s();
			else cg.f32x4.convert_i32x4_u();
		} else if (op === "Cmplt") {
			gen(src[0]), gen(src[1]);
			const srcDtype = src[0].dtype;
			if (srcDtype === "float32") cg.f32x4.lt();
			else if (srcDtype === "int32") cg.i32x4.lt_s();
			else if (srcDtype === "uint32") cg.i32x4.lt_u();
			else throw new UnsupportedOpError(op, dtype, "wasm");
			cg.i32.const(1);
			cg.i32x4.splat();
			cg.v128.and();
		} else if (op === "Cmpne") {
			gen(src[0]), gen(src[1]);
			if (src[0].dtype === "float32") cg.f32x4.ne();
			else cg.i32x4.ne();
			cg.i32.const(1);
			cg.i32x4.splat();
			cg.v128.and();
		} else if (op === "Where") {
			gen(src[1]);
			gen(src[2]);
			gen(src[0]);
			cg.i32.const(0);
			cg.i32x4.splat();
			cg.i32x4.ne();
			cg.v128.bitselect();
		} else if (op === "Variable" || op === "Special") throw new Error(`translateExpSimd: unexpected ${op}(${arg})`);
		else if (op === "GlobalIndex") {
			const [gid, len] = arg;
			const indexSubtree = src[0];
			const pointer = pointerMap.get(exp);
			const stride = pointer?.stride ?? strideMap.get(exp) ?? { kind: "gather" };
			if (pointer) {
				const cached = pointer.valueKey ? pointerValueCache.get(pointer.valueKey) : void 0;
				if (cached !== void 0) cg.local.get(cached);
				else {
					cg.local.get(pointer.ptr);
					if (stride.kind === "contiguous") if (isInt) cg.i32x4.load(4);
					else cg.f32x4.load(4);
					else if (stride.kind === "broadcast") if (isInt) {
						cg.i32.load(2);
						cg.i32x4.splat();
					} else {
						cg.f32.load(2);
						cg.f32x4.splat();
					}
					else throw new Error("reduction pointer plan cannot use gather loads");
					if (pointer.valueKey) {
						const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
						cg.local.tee(local);
						pointerValueCache.set(pointer.valueKey, local);
					}
				}
			} else if (stride.kind === "contiguous") {
				translateExp(cg, funcs, indexSubtree, ctx);
				{
					const maxIdx = Math.max(len - 4, 0);
					const wideIdx = cg.local.declare(cg.i32);
					cg.local.set(wideIdx);
					cg.local.get(wideIdx);
					cg.i32.const(maxIdx);
					cg.local.get(wideIdx);
					cg.i32.const(maxIdx);
					cg.i32.lt_u();
					cg.select();
				}
				cg.i32.const(byteWidth(dtype));
				cg.i32.mul();
				cg.local.get(gid);
				cg.i32.add();
				if (isInt) cg.i32x4.load(4);
				else cg.f32x4.load(4);
			} else if (stride.kind === "broadcast") {
				translateExp(cg, funcs, indexSubtree, ctx);
				const local = cg.local.declare(cg.i32);
				cg.local.tee(local);
				cg.i32.const(0);
				cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
				cg.select();
				cg.i32.const(byteWidth(dtype));
				cg.i32.mul();
				cg.local.get(gid);
				cg.i32.add();
				if (isInt) {
					cg.i32.load(2);
					cg.i32x4.splat();
				} else {
					cg.f32.load(2);
					cg.f32x4.splat();
				}
			} else {
				const steppingLocal = ctx["gidx"];
				const origValue = cg.local.declare(cg.i32);
				cg.local.get(steppingLocal);
				cg.local.set(origValue);
				if (isInt) {
					cg.i32.const(0);
					cg.i32x4.splat();
				} else {
					cg.f32.const(0);
					cg.f32x4.splat();
				}
				const vec = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
				cg.local.set(vec);
				const idx = cg.local.declare(cg.i32);
				const scalarVal = cg.local.declare(isInt ? cg.i32 : cg.f32);
				for (let lane = 0; lane < 4; lane++) {
					cg.local.get(origValue);
					if (lane > 0) {
						cg.i32.const(lane);
						cg.i32.add();
					}
					cg.local.set(steppingLocal);
					translateExp(cg, funcs, indexSubtree, ctx);
					cg.local.tee(idx);
					cg.i32.const(0);
					cg.local.get(idx), cg.i32.const(len), cg.i32.lt_u();
					cg.select();
					cg.i32.const(byteWidth(dtype));
					cg.i32.mul();
					cg.local.get(gid);
					cg.i32.add();
					if (isInt) cg.i32.load(2);
					else cg.f32.load(2);
					cg.local.set(scalarVal);
					cg.local.get(vec);
					cg.local.get(scalarVal);
					if (isInt) cg.i32x4.replace_lane(lane);
					else cg.f32x4.replace_lane(lane);
					cg.local.set(vec);
				}
				cg.local.get(origValue);
				cg.local.set(steppingLocal);
				cg.local.get(vec);
			}
		} else throw new Error(`translateExpSimd: unsupported op ${op}`);
		if ((references.get(exp) ?? 0) > 1) {
			const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
			cg.local.tee(local);
			expContext.set(exp, local);
		}
	};
	countReferences(exp);
	gen(exp);
}
function dty(cg, op, dtype) {
	switch (dtype) {
		case "float32": return cg.f32;
		case "float64": return cg.f64;
		case "int32":
		case "uint32":
		case "bool": return cg.i32;
		default: throw new UnsupportedOpError(op, dtype, "wasm");
	}
}
function dtyF(cg, op, dtype) {
	switch (dtype) {
		case "float32": return cg.f32;
		case "float64": return cg.f64;
		default: throw new UnsupportedOpError(op, dtype, "wasm");
	}
}
/** Subset of operations supported in SIMD compilation mode. */
const simdSupportedOps = /* @__PURE__ */ new Map();
simdSupportedOps.set("float32", /* @__PURE__ */ new Set([
	"Add",
	"Sub",
	"Mul",
	"Floor",
	"Ceil",
	"Min",
	"Max",
	"Sqrt",
	"Cast",
	"Where",
	"Const",
	"GlobalIndex"
]));
simdSupportedOps.set("int32", /* @__PURE__ */ new Set([
	"Add",
	"Sub",
	"Mul",
	"Min",
	"Max",
	"Cast",
	"Where",
	"Const",
	"GlobalIndex"
]));
simdSupportedOps.set("uint32", simdSupportedOps.get("int32"));
simdSupportedOps.set("bool", /* @__PURE__ */ new Set([
	"Add",
	"Mul",
	"Min",
	"Max",
	"Cmplt",
	"Cmpne",
	"Const",
	"GlobalIndex"
]));
//#endregion
//#region src/backend/wasm/wasmblr.ts
/**
* @file Minimalist WebAssembly assembler. This allows you to emit WebAssembly
* bytecode directly from the browser.
*
* Self-contained port of https://github.com/bwasti/wasmblr to TypeScript.
* Some operation names in this module are written in `snake_case` to match
* their names in the Wasm specification.
*
* Reference: https://pengowray.github.io/wasm-ops/.
*/
const magicModuleHeader = [
	0,
	97,
	115,
	109
];
const moduleVersion = [
	1,
	0,
	0,
	0
];
function assert(condition, message) {
	if (!condition) throw new Error(message || "Assertion failed");
}
function encodeSigned(n) {
	const out = [];
	let more = true;
	while (more) {
		let byte = n & 127;
		n >>= 7;
		if (n === 0 && (byte & 64) === 0 || n === -1 && (byte & 64) !== 0) more = false;
		else byte |= 128;
		out.push(byte);
	}
	return out;
}
function encodeUnsigned(n) {
	const out = [];
	do {
		let byte = n & 127;
		n = n >>> 7;
		if (n !== 0) byte |= 128;
		out.push(byte);
	} while (n !== 0);
	return out;
}
function encodeString(s) {
	const bytes = new TextEncoder().encode(s);
	return [bytes.length, ...bytes];
}
function encodeBlocktype(type) {
	assert(type.length > 0, "blocktype must have at least one type");
	if (type.length === 1) return [type[0].typeId];
	return [
		96,
		...encodeUnsigned(0),
		...encodeUnsigned(type.length),
		...type.map((t) => t.typeId)
	];
}
function encodeOpcode(opcode) {
	if (typeof opcode === "number") return [opcode];
	return [opcode[0], ...encodeUnsigned(opcode[1])];
}
function appendLengthEncodedBlock(out, inp) {
	out.push(...encodeUnsigned(inp.length));
	for (const b of inp) out.push(b);
}
var Function_ = class {
	inputTypes;
	outputTypes;
	body;
	locals = [];
	constructor(inputTypes, outputTypes, body) {
		this.inputTypes = inputTypes;
		this.outputTypes = outputTypes;
		this.body = body || (() => {});
	}
	emit() {
		this.locals = [];
		this.body();
	}
};
var Memory = class {
	cg;
	min = 0;
	max = 0;
	isShared = false;
	aString = "";
	bString = "";
	constructor(cg) {
		this.cg = cg;
	}
	/** Declare the size of the memory. Each page is 64 KiB. */
	pages(min, max = 0) {
		assert(this.min === 0 && this.max === 0);
		this.min = min;
		this.max = max;
		return this;
	}
	export(a) {
		assert(!this.isImport && !this.isExport, "already set");
		this.aString = a;
		return this;
	}
	shared(isShared) {
		this.isShared = isShared;
		return this;
	}
	import(a, b) {
		assert(!this.isImport && !this.isExport, "already set");
		this.aString = a;
		this.bString = b;
		return this;
	}
	size() {
		this.cg._emit(63);
		this.cg._emit(0);
	}
	grow() {
		this.cg._emit(64);
		this.cg._emit(0);
	}
	get isImport() {
		return this.aString.length > 0 && this.bString.length > 0;
	}
	get isExport() {
		return this.aString.length > 0 && this.bString.length === 0;
	}
};
/** Public API of WebAssembly assembler. */
var CodeGenerator = class {
	local;
	i32;
	f32;
	f64;
	v128;
	i32x4;
	f32x4;
	memory;
	void = {
		typeId: 64,
		name: "void"
	};
	#functions = [];
	#importedFunctions = [];
	#exportedFunctions = /* @__PURE__ */ new Map();
	#curFunction = null;
	#curBytes = [];
	#typeStack = [];
	#blockFrames = [];
	constructor() {
		this.local = new Local(this);
		this.i32 = new I32(this);
		this.f32 = new F32(this);
		this.f64 = new F64(this);
		this.v128 = new V128(this);
		this.i32x4 = new I32x4(this);
		this.f32x4 = new F32x4(this);
		this.memory = new Memory(this);
	}
	unreachable() {
		this._emit(0);
	}
	nop() {
		this._emit(1);
	}
	block(...type) {
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(2);
		this._emit(encodeBlocktype(type));
	}
	loop(...type) {
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(3);
		this._emit(encodeBlocktype(type));
	}
	if(...type) {
		assert(this._pop().typeId === this.i32.typeId, "if_: expected i32");
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(4);
		this._emit(encodeBlocktype(type));
	}
	else() {
		assert(this.#blockFrames.length > 0, "else: no block to else");
		const frame = this.#blockFrames[this.#blockFrames.length - 1];
		this.#typeStack = this.#typeStack.slice(0, frame.idx);
		this._emit(5);
	}
	/** End a block (`block`, `if`/`else`, `loop`, or function). */
	end() {
		const frame = this.#blockFrames.pop();
		assert(frame !== void 0, "end: no block to end");
		this.#typeStack = this.#typeStack.slice(0, frame.idx);
		for (const ty of frame.ty) if (ty.typeId !== this.void.typeId) this._push(ty);
		this._emit(11);
	}
	/** Branch to a block a certain depth outward on the stack. */
	br(depth) {
		this._emit(12);
		this._emit(encodeUnsigned(depth));
	}
	/** Conditional branch to a block a certain depth outward on the stack. */
	br_if(depth) {
		assert(this._pop().typeId === this.i32.typeId, "br_if: expected i32");
		this._emit(13);
		this._emit(encodeUnsigned(depth));
	}
	/** Jump table that indexes into a label vector (like switch). */
	br_table(...depths) {
		assert(this._pop().typeId === this.i32.typeId, "br_table: expected i32");
		assert(depths.length > 0, "br_table: expected at least one default depth");
		this._emit(14);
		this._emit(encodeUnsigned(depths.length - 1));
		for (const d of depths) this._emit(encodeUnsigned(d));
	}
	/** Return from a function, branching out of the outermost block. */
	return() {
		this._emit(15);
	}
	/** Call a function with the given ID. */
	call(fn) {
		assert(fn < this.#importedFunctions.length + this.#functions.length, "function index does not exist");
		const func = fn < this.#importedFunctions.length ? this.#importedFunctions[fn] : this.#functions[fn - this.#importedFunctions.length];
		for (let i = func.inputTypes.length - 1; i >= 0; i--) {
			const argType = this._pop();
			assert(argType.typeId === func.inputTypes[i].typeId, `call: argument ${i} type mismatch, expected ${func.inputTypes[i].name} got ${argType.name}`);
		}
		for (const outputType of func.outputTypes) this._push(outputType);
		this._emit(16);
		this._emit(encodeUnsigned(fn));
	}
	/** Throw away an operand on the stack. */
	drop() {
		this._pop();
		this._emit(26);
	}
	/** Select one of the first two operands (T, F) based on the third operand (i32)'s value. */
	select() {
		assert(this._pop().typeId === this.i32.typeId, "select: expected i32 condition");
		const [b, a] = [this._pop(), this._pop()];
		assert(a.typeId === b.typeId, "select: expected same type for both operands");
		this._push(a);
		this._emit(27);
	}
	/** Import a JavaScript function; returns its index. */
	importFunction(module, name, inputTypes, outputTypes) {
		if (this.#functions.length > 0) throw new Error("function imports must precede defining functions");
		const idx = this.#importedFunctions.length;
		this.#importedFunctions.push({
			module,
			name,
			inputTypes,
			outputTypes
		});
		return idx;
	}
	/** Export a function. */
	export(fn, name) {
		this.#exportedFunctions.set(fn, name);
	}
	/** Declare a new function; returns its index. */
	function(inputTypes, outputTypes, body) {
		const idx = this.#importedFunctions.length + this.#functions.length;
		this.#functions.push(new Function_(inputTypes, outputTypes, body));
		return idx;
	}
	_declareLocal(type) {
		assert(this.#curFunction !== null, "No current function");
		const idx = this.#curFunction.locals.length + this.#curFunction.inputTypes.length;
		this.#curFunction.locals.push(type);
		return idx;
	}
	_inputTypes() {
		assert(this.#curFunction !== null, "No current function");
		return this.#curFunction.inputTypes;
	}
	_locals() {
		assert(this.#curFunction !== null, "No current function");
		return this.#curFunction.locals;
	}
	_push(type) {
		if (!type) throw new Error(`pushing type ${type}`);
		this.#typeStack.push(type);
	}
	_pop() {
		assert(this.#typeStack.length > 0, "popping empty stack");
		return this.#typeStack.pop();
	}
	_emit(bytes) {
		if (typeof bytes === "number") this.#curBytes.push(bytes);
		else this.#curBytes.push(...bytes);
	}
	finish() {
		this.#curBytes = [];
		const emittedBytes = [];
		emittedBytes.push(...magicModuleHeader);
		emittedBytes.push(...moduleVersion);
		const typeSectionBytes = [];
		const totalFunctionTypes = this.#importedFunctions.length + this.#functions.length;
		typeSectionBytes.push(...encodeUnsigned(totalFunctionTypes));
		for (const f of [...this.#importedFunctions, ...this.#functions]) {
			typeSectionBytes.push(96);
			typeSectionBytes.push(...encodeUnsigned(f.inputTypes.length));
			for (const t of f.inputTypes) typeSectionBytes.push(t.typeId);
			typeSectionBytes.push(...encodeUnsigned(f.outputTypes.length));
			for (const t of f.outputTypes) typeSectionBytes.push(t.typeId);
		}
		emittedBytes.push(1);
		appendLengthEncodedBlock(emittedBytes, typeSectionBytes);
		const importSectionBytes = [];
		const numImports = this.#importedFunctions.length + (this.memory.isImport ? 1 : 0);
		if (numImports > 0) {
			importSectionBytes.push(...encodeUnsigned(numImports));
			for (let i = 0; i < this.#importedFunctions.length; i++) {
				const f = this.#importedFunctions[i];
				importSectionBytes.push(...encodeString(f.module));
				importSectionBytes.push(...encodeString(f.name));
				importSectionBytes.push(0);
				importSectionBytes.push(...encodeUnsigned(i));
			}
			if (this.memory.isImport) {
				importSectionBytes.push(...encodeString(this.memory.aString));
				importSectionBytes.push(...encodeString(this.memory.bString));
				importSectionBytes.push(2);
				if (this.memory.max) {
					if (this.memory.isShared) importSectionBytes.push(3);
					else importSectionBytes.push(1);
					importSectionBytes.push(...encodeUnsigned(this.memory.min));
					importSectionBytes.push(...encodeUnsigned(this.memory.max));
				} else {
					assert(!this.memory.isShared, "shared memory must have a max size");
					importSectionBytes.push(0);
					importSectionBytes.push(...encodeUnsigned(this.memory.min));
				}
			}
			emittedBytes.push(2);
			appendLengthEncodedBlock(emittedBytes, importSectionBytes);
		}
		const functionSectionBytes = [];
		functionSectionBytes.push(...encodeUnsigned(this.#functions.length));
		for (let i = 0; i < this.#functions.length; i++) {
			const typeIndex = this.#importedFunctions.length + i;
			functionSectionBytes.push(...encodeUnsigned(typeIndex));
		}
		emittedBytes.push(3);
		appendLengthEncodedBlock(emittedBytes, functionSectionBytes);
		const memorySectionBytes = [];
		if (!this.memory.isImport && (this.memory.min || this.memory.max)) {
			memorySectionBytes.push(1);
			if (this.memory.min && this.memory.max) {
				if (this.memory.isShared) memorySectionBytes.push(3);
				else memorySectionBytes.push(1);
				memorySectionBytes.push(...encodeUnsigned(this.memory.min));
				memorySectionBytes.push(...encodeUnsigned(this.memory.max));
			} else {
				assert(!this.memory.isShared, "shared memory must have a max size");
				memorySectionBytes.push(0);
				memorySectionBytes.push(...encodeUnsigned(this.memory.min));
			}
			emittedBytes.push(5);
			appendLengthEncodedBlock(emittedBytes, memorySectionBytes);
		}
		const exportSectionBytes = [];
		const numExports = this.#exportedFunctions.size + (this.memory.isExport ? 1 : 0);
		exportSectionBytes.push(...encodeUnsigned(numExports));
		if (this.memory.isExport) {
			exportSectionBytes.push(...encodeString(this.memory.aString));
			exportSectionBytes.push(2);
			exportSectionBytes.push(0);
		}
		for (const [key, name] of this.#exportedFunctions.entries()) {
			exportSectionBytes.push(...encodeString(name));
			exportSectionBytes.push(0);
			exportSectionBytes.push(...encodeUnsigned(key));
		}
		emittedBytes.push(7);
		appendLengthEncodedBlock(emittedBytes, exportSectionBytes);
		const codeSectionBytes = [];
		codeSectionBytes.push(...encodeUnsigned(this.#functions.length));
		for (const f of this.#functions) {
			this.#typeStack = [];
			this.#blockFrames = [{
				idx: 0,
				ty: f.outputTypes
			}];
			this.#curFunction = f;
			this.#curBytes = [];
			f.emit();
			this.end();
			const bodyBytes = this.#curBytes;
			this.#curBytes = [];
			this.#curBytes.push(...encodeUnsigned(f.locals.length));
			for (const l of f.locals) {
				this._emit(1);
				this._emit(l.typeId);
			}
			appendLengthEncodedBlock(codeSectionBytes, this.#curBytes.concat(bodyBytes));
		}
		this.#curFunction = null;
		emittedBytes.push(10);
		appendLengthEncodedBlock(emittedBytes, codeSectionBytes);
		return new Uint8Array(emittedBytes);
	}
};
var Local = class {
	cg;
	constructor(cg) {
		this.cg = cg;
	}
	declare(type) {
		return this.cg._declareLocal(type);
	}
	get(idx) {
		assert(Number.isInteger(idx), "getting non-integer local");
		const inputTypes = this.cg._inputTypes();
		if (idx < inputTypes.length) this.cg._push(inputTypes[idx]);
		else this.cg._push(this.cg._locals()[idx - inputTypes.length]);
		this.cg._emit(32);
		this.cg._emit(encodeUnsigned(idx));
	}
	set(idx) {
		const t = this.cg._pop();
		const inputTypes = this.cg._inputTypes();
		assert((idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length]).typeId === t.typeId, "can't set local to this value (wrong type)");
		this.cg._emit(33);
		this.cg._emit(encodeUnsigned(idx));
	}
	tee(idx) {
		const t = this.cg._pop();
		const inputTypes = this.cg._inputTypes();
		const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
		assert(expectedType.typeId === t.typeId, "can't tee local to this value (wrong type)");
		this.cg._emit(34);
		this.cg._emit(encodeUnsigned(idx));
		this.cg._push(expectedType);
	}
};
function UNARY_OP(op, opcode, inType, outType) {
	return function() {
		assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inType} -> ${outType})`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._push(this.cg[outType]);
	};
}
function BINARY_OP(op, opcode, typeA, typeB, outType) {
	return function() {
		const b = this.cg._pop();
		assert(this.cg._pop().typeId === this.cg[typeA].typeId && b.typeId === this.cg[typeB].typeId, `invalid type for ${op} (${typeA}, ${typeB} -> ${outType})`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._push(this.cg[outType]);
	};
}
function LOAD_OP(op, opcode, outType) {
	return function(align = 0, offset = 0) {
		assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
		this.cg._push(this.cg[outType]);
	};
}
function STORE_OP(op, opcode, inType) {
	return function(align = 0, offset = 0) {
		const valType = this.cg._pop();
		const idxType = this.cg._pop();
		assert(valType.typeId === this.cg[inType].typeId, `invalid value type for ${op} (${inType})`);
		assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
	};
}
var I32 = class {
	cg;
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 127;
	}
	get name() {
		return "i32";
	}
	const(i) {
		this.cg._emit(65);
		this.cg._emit(encodeSigned(i));
		this.cg._push(this);
	}
	clz = UNARY_OP("clz", 103, "i32", "i32");
	ctz = UNARY_OP("ctz", 104, "i32", "i32");
	popcnt = UNARY_OP("popcnt", 105, "i32", "i32");
	lt_s = BINARY_OP("lt_s", 72, "i32", "i32", "i32");
	lt_u = BINARY_OP("lt_u", 73, "i32", "i32", "i32");
	gt_s = BINARY_OP("gt_s", 74, "i32", "i32", "i32");
	gt_u = BINARY_OP("gt_u", 75, "i32", "i32", "i32");
	le_s = BINARY_OP("le_s", 76, "i32", "i32", "i32");
	le_u = BINARY_OP("le_u", 77, "i32", "i32", "i32");
	ge_s = BINARY_OP("ge_s", 78, "i32", "i32", "i32");
	ge_u = BINARY_OP("ge_u", 79, "i32", "i32", "i32");
	add = BINARY_OP("add", 106, "i32", "i32", "i32");
	sub = BINARY_OP("sub", 107, "i32", "i32", "i32");
	mul = BINARY_OP("mul", 108, "i32", "i32", "i32");
	div_s = BINARY_OP("div_s", 109, "i32", "i32", "i32");
	div_u = BINARY_OP("div_u", 110, "i32", "i32", "i32");
	rem_s = BINARY_OP("rem_s", 111, "i32", "i32", "i32");
	rem_u = BINARY_OP("rem_u", 112, "i32", "i32", "i32");
	and = BINARY_OP("and", 113, "i32", "i32", "i32");
	or = BINARY_OP("or", 114, "i32", "i32", "i32");
	xor = BINARY_OP("xor", 115, "i32", "i32", "i32");
	shl = BINARY_OP("shl", 116, "i32", "i32", "i32");
	shr_s = BINARY_OP("shr_s", 117, "i32", "i32", "i32");
	shr_u = BINARY_OP("shr_u", 118, "i32", "i32", "i32");
	rotl = BINARY_OP("rotl", 119, "i32", "i32", "i32");
	rotr = BINARY_OP("rotr", 120, "i32", "i32", "i32");
	eqz = UNARY_OP("eqz", 69, "i32", "i32");
	eq = BINARY_OP("eq", 70, "i32", "i32", "i32");
	ne = BINARY_OP("ne", 71, "i32", "i32", "i32");
	trunc_f32_s = UNARY_OP("trunc_f32_s", 168, "f32", "i32");
	trunc_f32_u = UNARY_OP("trunc_f32_u", 169, "f32", "i32");
	trunc_f64_s = UNARY_OP("trunc_f64_s", 170, "f64", "i32");
	trunc_f64_u = UNARY_OP("trunc_f64_u", 171, "f64", "i32");
	load = LOAD_OP("load", 40, "i32");
	load8_s = LOAD_OP("load8_s", 44, "i32");
	load8_u = LOAD_OP("load8_u", 45, "i32");
	load16_s = LOAD_OP("load16_s", 46, "i32");
	load16_u = LOAD_OP("load16_u", 47, "i32");
	store = STORE_OP("store", 54, "i32");
	store8 = STORE_OP("store8", 58, "i32");
	store16 = STORE_OP("store16", 59, "i32");
	reinterpret_f32 = UNARY_OP("reinterpret_f32", 188, "f32", "i32");
	trunc_sat_f32_s = UNARY_OP("trunc_sat_f32_s", [252, 0], "f32", "i32");
	trunc_sat_f32_u = UNARY_OP("trunc_sat_f32_u", [252, 1], "f32", "i32");
	trunc_sat_f64_s = UNARY_OP("trunc_sat_f64_s", [252, 2], "f64", "i32");
	trunc_sat_f64_u = UNARY_OP("trunc_sat_f64_u", [252, 3], "f64", "i32");
};
var F32 = class {
	cg;
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 125;
	}
	get name() {
		return "f32";
	}
	const(f) {
		this.cg._emit(67);
		const buffer = /* @__PURE__ */ new ArrayBuffer(4);
		new DataView(buffer).setFloat32(0, f, true);
		const bytes = new Uint8Array(buffer);
		for (let i = 0; i < 4; i++) this.cg._emit(bytes[i]);
		this.cg._push(this);
	}
	load = LOAD_OP("load", 42, "f32");
	store = STORE_OP("store", 56, "f32");
	eq = BINARY_OP("eq", 91, "f32", "f32", "i32");
	ne = BINARY_OP("ne", 92, "f32", "f32", "i32");
	lt = BINARY_OP("lt", 93, "f32", "f32", "i32");
	gt = BINARY_OP("gt", 94, "f32", "f32", "i32");
	le = BINARY_OP("le", 95, "f32", "f32", "i32");
	ge = BINARY_OP("ge", 96, "f32", "f32", "i32");
	abs = UNARY_OP("abs", 139, "f32", "f32");
	neg = UNARY_OP("neg", 140, "f32", "f32");
	ceil = UNARY_OP("ceil", 141, "f32", "f32");
	floor = UNARY_OP("floor", 142, "f32", "f32");
	trunc = UNARY_OP("trunc", 143, "f32", "f32");
	nearest = UNARY_OP("nearest", 144, "f32", "f32");
	sqrt = UNARY_OP("sqrt", 145, "f32", "f32");
	add = BINARY_OP("add", 146, "f32", "f32", "f32");
	sub = BINARY_OP("sub", 147, "f32", "f32", "f32");
	mul = BINARY_OP("mul", 148, "f32", "f32", "f32");
	div = BINARY_OP("div", 149, "f32", "f32", "f32");
	min = BINARY_OP("min", 150, "f32", "f32", "f32");
	max = BINARY_OP("max", 151, "f32", "f32", "f32");
	copysign = BINARY_OP("copysign", 152, "f32", "f32", "f32");
	convert_i32_s = UNARY_OP("convert_i32_s", 178, "i32", "f32");
	convert_i32_u = UNARY_OP("convert_i32_u", 179, "i32", "f32");
	demote_f64 = UNARY_OP("demote_f64", 182, "f64", "f32");
	reinterpret_i32 = UNARY_OP("reinterpret_i32", 190, "i32", "f32");
};
var F64 = class {
	cg;
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 124;
	}
	get name() {
		return "f64";
	}
	const(f) {
		this.cg._emit(68);
		const buffer = /* @__PURE__ */ new ArrayBuffer(8);
		new DataView(buffer).setFloat64(0, f, true);
		const bytes = new Uint8Array(buffer);
		for (let i = 0; i < 8; i++) this.cg._emit(bytes[i]);
		this.cg._push(this);
	}
	load = LOAD_OP("load", 43, "f64");
	store = STORE_OP("store", 57, "f64");
	eq = BINARY_OP("eq", 97, "f64", "f64", "i32");
	ne = BINARY_OP("ne", 98, "f64", "f64", "i32");
	lt = BINARY_OP("lt", 99, "f64", "f64", "i32");
	gt = BINARY_OP("gt", 100, "f64", "f64", "i32");
	le = BINARY_OP("le", 101, "f64", "f64", "i32");
	ge = BINARY_OP("ge", 102, "f64", "f64", "i32");
	abs = UNARY_OP("abs", 153, "f64", "f64");
	neg = UNARY_OP("neg", 154, "f64", "f64");
	ceil = UNARY_OP("ceil", 155, "f64", "f64");
	floor = UNARY_OP("floor", 156, "f64", "f64");
	trunc = UNARY_OP("trunc", 157, "f64", "f64");
	nearest = UNARY_OP("nearest", 158, "f64", "f64");
	sqrt = UNARY_OP("sqrt", 159, "f64", "f64");
	add = BINARY_OP("add", 160, "f64", "f64", "f64");
	sub = BINARY_OP("sub", 161, "f64", "f64", "f64");
	mul = BINARY_OP("mul", 162, "f64", "f64", "f64");
	div = BINARY_OP("div", 163, "f64", "f64", "f64");
	min = BINARY_OP("min", 164, "f64", "f64", "f64");
	max = BINARY_OP("max", 165, "f64", "f64", "f64");
	copysign = BINARY_OP("copysign", 166, "f64", "f64", "f64");
	convert_i32_s = UNARY_OP("convert_i32_s", 183, "i32", "f64");
	convert_i32_u = UNARY_OP("convert_i32_u", 184, "i32", "f64");
	promote_f32 = UNARY_OP("promote_f32", 187, "f32", "f64");
};
function VECTOR_OP(op, vopcode, inTypes, outType) {
	return function() {
		for (const inType of inTypes.toReversed()) assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes.join(", ")} -> ${outType})`);
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._push(this.cg[outType]);
	};
}
function VECTOR_OPL(op, vopcode, inTypes, outType) {
	return function(lane) {
		for (const inType of inTypes.toReversed()) assert(this.cg._pop().typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes} -> ${outType})`);
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._emit(lane);
		this.cg._push(this.cg[outType]);
	};
}
function VECTOR_LOAD_OP(op, vopcode) {
	return function(align = 0, offset = 0) {
		assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
		this.cg._push(this.cg.v128);
	};
}
var V128 = class {
	cg;
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 123;
	}
	get name() {
		return "v128";
	}
	load = VECTOR_LOAD_OP("load", 0);
	load32x2_s = VECTOR_LOAD_OP("load32x2_s", 5);
	load32x2_u = VECTOR_LOAD_OP("load32x2_u", 6);
	load32_splat = VECTOR_LOAD_OP("load32_splat", 9);
	load32_zero = VECTOR_LOAD_OP("load32_zero", 92);
	store(align = 0, offset = 0) {
		assert(this.cg._pop().typeId === this.cg.v128.typeId, `invalid type for store`);
		assert(this.cg._pop().typeId === this.cg.i32.typeId, `invalid type for store`);
		this.cg._emit(253);
		this.cg._emit(encodeUnsigned(11));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
	}
	not = VECTOR_OP("not", 77, ["v128"], "v128");
	and = VECTOR_OP("and", 78, ["v128", "v128"], "v128");
	andnot = VECTOR_OP("andnot", 79, ["v128", "v128"], "v128");
	or = VECTOR_OP("or", 80, ["v128", "v128"], "v128");
	xor = VECTOR_OP("xor", 81, ["v128", "v128"], "v128");
	bitselect = VECTOR_OP("bitselect", 82, [
		"v128",
		"v128",
		"v128"
	], "v128");
	any_true = VECTOR_OP("any_true", 83, ["v128"], "i32");
};
var I32x4 = class extends V128 {
	splat = VECTOR_OP("splat", 17, ["i32"], "v128");
	extract_lane = VECTOR_OPL("extract_lane", 27, ["v128"], "i32");
	replace_lane = VECTOR_OPL("replace_lane", 28, ["v128", "i32"], "v128");
	eq = VECTOR_OP("eq", 55, ["v128", "v128"], "v128");
	ne = VECTOR_OP("ne", 56, ["v128", "v128"], "v128");
	lt_s = VECTOR_OP("lt_s", 57, ["v128", "v128"], "v128");
	lt_u = VECTOR_OP("lt_u", 58, ["v128", "v128"], "v128");
	gt_s = VECTOR_OP("gt_s", 59, ["v128", "v128"], "v128");
	gt_u = VECTOR_OP("gt_u", 60, ["v128", "v128"], "v128");
	le_s = VECTOR_OP("le_s", 61, ["v128", "v128"], "v128");
	le_u = VECTOR_OP("le_u", 62, ["v128", "v128"], "v128");
	ge_s = VECTOR_OP("ge_s", 63, ["v128", "v128"], "v128");
	ge_u = VECTOR_OP("ge_u", 64, ["v128", "v128"], "v128");
	abs = VECTOR_OP("abs", 160, ["v128"], "v128");
	neg = VECTOR_OP("neg", 161, ["v128"], "v128");
	all_true = VECTOR_OP("all_true", 163, ["v128"], "i32");
	bitmask = VECTOR_OP("bitmask", 164, ["v128"], "i32");
	shl = VECTOR_OP("shl", 171, ["v128", "i32"], "v128");
	shr_s = VECTOR_OP("shr_s", 172, ["v128", "i32"], "v128");
	shr_u = VECTOR_OP("shr_u", 173, ["v128", "i32"], "v128");
	add = VECTOR_OP("add", 174, ["v128", "v128"], "v128");
	sub = VECTOR_OP("sub", 177, ["v128", "v128"], "v128");
	mul = VECTOR_OP("mul", 181, ["v128", "v128"], "v128");
	min_s = VECTOR_OP("min_s", 182, ["v128", "v128"], "v128");
	min_u = VECTOR_OP("min_u", 183, ["v128", "v128"], "v128");
	max_s = VECTOR_OP("max_s", 184, ["v128", "v128"], "v128");
	max_u = VECTOR_OP("max_u", 185, ["v128", "v128"], "v128");
	trunc_sat_f32x4_s = VECTOR_OP("trunc_sat_f32x4_s", 248, ["v128"], "v128");
	trunc_sat_f32x4_u = VECTOR_OP("trunc_sat_f32x4_u", 249, ["v128"], "v128");
};
var F32x4 = class extends V128 {
	splat = VECTOR_OP("splat", 19, ["f32"], "v128");
	extract_lane = VECTOR_OPL("extract_lane", 31, ["v128"], "f32");
	replace_lane = VECTOR_OPL("replace_lane", 32, ["v128", "f32"], "v128");
	eq = VECTOR_OP("eq", 65, ["v128", "v128"], "v128");
	ne = VECTOR_OP("ne", 66, ["v128", "v128"], "v128");
	lt = VECTOR_OP("lt", 67, ["v128", "v128"], "v128");
	gt = VECTOR_OP("gt", 68, ["v128", "v128"], "v128");
	le = VECTOR_OP("le", 69, ["v128", "v128"], "v128");
	ge = VECTOR_OP("ge", 70, ["v128", "v128"], "v128");
	ceil = VECTOR_OP("ceil", 103, ["v128"], "v128");
	floor = VECTOR_OP("floor", 104, ["v128"], "v128");
	trunc = VECTOR_OP("trunc", 105, ["v128"], "v128");
	nearest = VECTOR_OP("nearest", 106, ["v128"], "v128");
	abs = VECTOR_OP("abs", 224, ["v128"], "v128");
	neg = VECTOR_OP("neg", 225, ["v128"], "v128");
	sqrt = VECTOR_OP("sqrt", 227, ["v128"], "v128");
	add = VECTOR_OP("add", 228, ["v128", "v128"], "v128");
	sub = VECTOR_OP("sub", 229, ["v128", "v128"], "v128");
	mul = VECTOR_OP("mul", 230, ["v128", "v128"], "v128");
	div = VECTOR_OP("div", 231, ["v128", "v128"], "v128");
	min = VECTOR_OP("min", 232, ["v128", "v128"], "v128");
	max = VECTOR_OP("max", 233, ["v128", "v128"], "v128");
	pmin = VECTOR_OP("pmin", 234, ["v128", "v128"], "v128");
	pmax = VECTOR_OP("pmax", 235, ["v128", "v128"], "v128");
	convert_i32x4_s = VECTOR_OP("convert_i32x4_s", 250, ["v128"], "v128");
	convert_i32x4_u = VECTOR_OP("convert_i32x4_u", 251, ["v128"], "v128");
	relaxed_madd = VECTOR_OP("relaxed_madd", 261, [
		"v128",
		"v128",
		"v128"
	], "v128");
	relaxed_nmadd = VECTOR_OP("relaxed_nmadd", 262, [
		"v128",
		"v128",
		"v128"
	], "v128");
};
//#endregion
//#region src/backend/wasm/codegen.ts
function isIdentityEpilogue(exp) {
	return exp?.op === "Variable" && exp.arg === "acc";
}
function initializeReductionPointer(cg, funcs, candidate, ctx, valueKey, ridxOffset) {
	const ptr = cg.local.declare(cg.i32);
	translateExp(cg, funcs, candidate.baseIndex, ctx);
	cg.i32.const(byteWidth(candidate.dtype));
	cg.i32.mul();
	cg.local.get(candidate.gid);
	cg.i32.add();
	if (ridxOffset !== void 0 && candidate.strideBytes !== 0) {
		cg.local.get(ridxOffset);
		cg.i32.const(candidate.strideBytes);
		cg.i32.mul();
		cg.i32.add();
	}
	cg.local.set(ptr);
	return {
		...candidate,
		ptr,
		valueKey
	};
}
function incrementReductionPointers(cg, pointers, multiplier = 1) {
	for (const pointer of pointers) {
		if (pointer.strideBytes === 0) continue;
		cg.local.get(pointer.ptr);
		cg.i32.const(pointer.strideBytes * multiplier);
		cg.i32.add();
		cg.local.set(pointer.ptr);
	}
}
function emitSimdReductionOp(cg, re, reIsInt, valueAlreadyAccumulated) {
	if (!reIsInt && valueAlreadyAccumulated) return;
	switch (re.op) {
		case "Add":
			if (reIsInt) cg.i32x4.add();
			else cg.f32x4.add();
			return;
		case "Mul":
			if (reIsInt) cg.i32x4.mul();
			else cg.f32x4.mul();
			return;
		case "Min":
			if (reIsInt) if (re.dtype === "int32") cg.i32x4.min_s();
			else cg.i32x4.min_u();
			else cg.f32x4.min();
			return;
		case "Max":
			if (reIsInt) if (re.dtype === "int32") cg.i32x4.max_s();
			else cg.i32x4.max_u();
			else cg.f32x4.max();
			return;
		default: throw new Error(`invalid SIMD reduction op: ${re.op}`);
	}
}
function emitScalarReductionOp(cg, re, acc) {
	switch (re.op) {
		case "Add":
			cg.local.get(acc);
			if (re.dtype === "bool") cg.i32.or();
			else dty(cg, re.op, re.dtype).add();
			return;
		case "Mul":
			cg.local.get(acc);
			if (re.dtype === "bool") cg.i32.and();
			else dty(cg, re.op, re.dtype).mul();
			return;
		case "Min":
		case "Max":
			if (isFloatDtype(re.dtype)) {
				cg.local.get(acc);
				if (re.op === "Min") dtyF(cg, re.op, re.dtype).min();
				else dtyF(cg, re.op, re.dtype).max();
			} else if ([
				"int32",
				"uint32",
				"bool"
			].includes(re.dtype)) {
				const local = cg.local.declare(cg.i32);
				cg.local.tee(local);
				cg.local.get(acc);
				cg.local.get(local);
				cg.local.get(acc);
				if (re.op === "Min") if (re.dtype === "int32") cg.i32.lt_s();
				else cg.i32.lt_u();
				else if (re.dtype === "int32") cg.i32.gt_s();
				else cg.i32.gt_u();
				cg.select();
			} else throw new Error(`invalid reduction min/max over ${re.dtype}`);
			return;
		default: throw new Error(`invalid wasm reduction op: ${re.op}`);
	}
}
/**
* Check if a kernel is eligible for SIMD codegen.
*
* A kernel qualifies when:
* - size >= 4 (need at least 4 elements for a SIMD group)
* - For reductions: the reduction op has a SIMD variant for its dtype
* - All nodes have a supported dtype (f32, i32, u32, bool) with SIMD variants
*/
function isSimdEligible(tunedExp, kernel) {
	if (kernel.size < 4) return false;
	if (kernel.reduction) {
		if (!simdSupportedOps.get(kernel.reduction.dtype)?.has(kernel.reduction.op)) return false;
	}
	const check = (exp, visited) => {
		if (visited.has(exp)) return true;
		visited.add(exp);
		if (!simdSupportedOps.get(exp.dtype)?.has(exp.op)) return false;
		if (exp.op === "GlobalIndex") return true;
		for (const child of exp.src) if (!check(child, visited)) return false;
		return true;
	};
	return check(tunedExp, /* @__PURE__ */ new Set());
}
/** Checks if SIMD over the reduced-k dimension is workable. */
function canUseKSimdReduction(exp, re, pointers) {
	if (re.op !== "Add") return false;
	return exp.collect((node) => node.op === "GlobalIndex").length === pointers.length && pointers.every((candidate) => candidate.dtype === "float32" && (candidate.strideBytes === 0 || candidate.strideBytes === byteWidth(candidate.dtype)));
}
/** Emit a runtime guard: enter the if-block only when [begin, end) is SIMD-aligned. */
function emitAlignmentGuard(cg, paramBegin, paramEnd, alignment = 4) {
	cg.local.get(paramEnd);
	cg.local.get(paramBegin);
	cg.i32.sub();
	cg.i32.const(alignment);
	cg.i32.rem_u();
	cg.i32.eqz();
	cg.local.get(paramBegin);
	cg.i32.const(alignment);
	cg.i32.rem_u();
	cg.i32.eqz();
	cg.i32.and();
	cg.if(cg.void);
}
function codegenWasm(kernel) {
	const tune = tuneNullopt(kernel);
	const re = kernel.reduction;
	if (DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
	const simdEligible = isSimdEligible(tune.exp, kernel);
	const hasIdentityEpilogue = isIdentityEpilogue(tune.epilogue);
	const expStrides = simdEligible ? collectSimdStrides(tune.exp) : /* @__PURE__ */ new Map();
	const reductionHasLaneGather = re && [...expStrides.values()].some((stride) => stride.kind === "gather");
	const useSimd = simdEligible && !reductionHasLaneGather;
	const simdReductionPointerCandidates = useSimd && re ? reductionPointerCandidates(tune.exp, expStrides) : [];
	const reductionPointers = re ? reductionPointerCandidates(tune.exp) : [];
	const useKSimdReduction = simdEligible && re && canUseKSimdReduction(tune.exp, re, reductionPointers);
	const kSimdReductionPointerCandidates = useKSimdReduction && re ? reductionPointers.map((candidate) => ({
		...candidate,
		stride: candidate.strideBytes === 0 ? {
			kind: "broadcast",
			tileSize: Infinity
		} : {
			kind: "contiguous",
			tileSize: Infinity
		}
	})) : [];
	const canStoreSimdPartials = hasIdentityEpilogue || re !== void 0 && kernel.dtype === re.dtype && byteWidth(kernel.dtype) === 4;
	const simdTilePlan = useSimd && re ? reductionTilePlan(kernel, expStrides, canStoreSimdPartials) : null;
	const kSimdTilePlan = reductionHasLaneGather && useKSimdReduction ? reductionKTilePlan(kernel, expStrides) : null;
	const useRelaxedMadd = hasWasmFeature("relaxed-madd") && re?.op === "Add" && tune.exp.dtype === "float32" && tune.exp.op === "Mul";
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	if (hasSharedArrayBuffer()) cg.memory.pages(0, 65536).shared(true);
	const distinctOps = mapSetUnion(tune.exp.distinctOps(), tune.epilogue?.distinctOps());
	const funcs = {};
	if (distinctOps.has("Sin")) funcs.sin = wasm_sin(cg);
	if (distinctOps.has("Cos")) funcs.cos = wasm_cos(cg);
	if (distinctOps.has("Asin")) funcs.asin = wasm_asin(cg);
	if (distinctOps.has("Atan")) funcs.atan = wasm_atan(cg);
	if (distinctOps.has("Exp") || distinctOps.has("Erf") || distinctOps.has("Erfc")) funcs.exp = wasm_exp(cg);
	if (distinctOps.has("Log")) funcs.log = wasm_log(cg);
	if (distinctOps.has("Erf")) funcs.erf = wasm_erf(cg, funcs.exp);
	if (distinctOps.has("Erfc")) funcs.erfc = wasm_erfc(cg, funcs.exp);
	if (distinctOps.has("Threefry2x32")) funcs.threefry2x32 = wasm_threefry2x32(cg);
	const paramBegin = kernel.nargs + 1;
	const paramEnd = kernel.nargs + 2;
	const kernelFunc = cg.function(rep(kernel.nargs + 3, cg.i32), [], () => {
		const gidx = cg.local.declare(cg.i32);
		cg.local.get(paramBegin);
		cg.local.set(gidx);
		const emitLocalPlusConst = (local, amount) => {
			cg.local.get(local);
			cg.i32.const(amount);
			cg.i32.add();
		};
		const bumpLocal = (local, amount) => {
			emitLocalPlusConst(local, amount);
			cg.local.set(local);
		};
		const setLocalConst = (local, value) => {
			cg.i32.const(value);
			cg.local.set(local);
		};
		const copyLocal = (target, source) => {
			cg.local.get(source);
			cg.local.set(target);
		};
		const setRowBase = (target, rowTileBase, rowOffset, tileSize) => {
			cg.local.get(rowTileBase);
			cg.local.get(rowOffset);
			cg.i32.const(tileSize);
			cg.i32.mul();
			cg.i32.add();
			cg.local.set(target);
		};
		const emitOutputAddress = (index) => {
			cg.local.get(kernel.nargs);
			cg.local.get(index);
			cg.i32.const(byteWidth(kernel.dtype));
			cg.i32.mul();
			cg.i32.add();
		};
		const declareTileGidx = (rowBase, col, rowOffset, colOffset) => {
			const local = cg.local.declare(cg.i32);
			cg.local.get(rowBase);
			if (rowOffset !== 0) {
				cg.i32.const(rowOffset);
				cg.i32.add();
			}
			cg.local.get(col);
			cg.i32.add();
			if (colOffset !== 0) {
				cg.i32.const(colOffset);
				cg.i32.add();
			}
			cg.local.set(local);
			return local;
		};
		const emitLoopWithBreaks = (emitBreaks, emitBody) => {
			cg.loop(cg.void);
			cg.block(cg.void);
			emitBreaks();
			emitBody();
			cg.br(1);
			cg.end();
			cg.end();
		};
		const emitLoopWhileLt = (index, emitBound, emitBody) => emitLoopWithBreaks(() => {
			cg.local.get(index);
			emitBound();
			cg.i32.ge_u();
			cg.br_if(0);
		}, emitBody);
		const emitLoopWhileLtAndConstLt = (index, emitBound, constBound, emitBody) => emitLoopWithBreaks(() => {
			cg.local.get(index);
			emitBound();
			cg.i32.ge_u();
			cg.br_if(0);
			cg.local.get(index);
			cg.i32.const(constBound);
			cg.i32.ge_u();
			cg.br_if(0);
		}, emitBody);
		const emitLoopWhileBlockFits = (index, blockSize, emitBound, constBound, emitBody) => emitLoopWithBreaks(() => {
			emitLocalPlusConst(index, blockSize);
			emitBound();
			cg.i32.gt_u();
			cg.br_if(0);
			emitLocalPlusConst(index, blockSize);
			cg.i32.const(constBound);
			cg.i32.gt_u();
			cg.br_if(0);
		}, emitBody);
		const emitRowTileLoop = (rowOffset, rowTileBase, tileRows, tileSize, emitBody) => emitLoopWithBreaks(() => {
			cg.local.get(rowOffset);
			cg.i32.const(tileRows);
			cg.i32.ge_u();
			cg.br_if(0);
			cg.local.get(rowTileBase);
			cg.local.get(rowOffset);
			cg.i32.const(tileSize);
			cg.i32.mul();
			cg.i32.add();
			cg.local.get(paramEnd);
			cg.i32.ge_u();
			cg.br_if(0);
		}, emitBody);
		const emitLoopWhileLocalLt = (index, bound, emitBody) => emitLoopWhileLt(index, () => cg.local.get(bound), emitBody);
		const emitLoopWhileConstLt = (index, bound, emitBody) => emitLoopWhileLt(index, () => cg.i32.const(bound), emitBody);
		const emitSimdExpWithAccumulator = (ctx, pointerMap, pointerValueCache, acc) => {
			if (useRelaxedMadd) {
				translateExpSimd(cg, funcs, tune.exp.src[0], ctx, expStrides, pointerMap, pointerValueCache);
				translateExpSimd(cg, funcs, tune.exp.src[1], ctx, expStrides, pointerMap, pointerValueCache);
				cg.local.get(acc);
				cg.f32x4.relaxed_madd();
				return true;
			}
			translateExpSimd(cg, funcs, tune.exp, ctx, expStrides, pointerMap, pointerValueCache);
			cg.local.get(acc);
			return false;
		};
		const emitSimdReductionForGidxs = (gidxs, pointerMaps, uniquePointers, ridxStart, ridxEnd) => {
			if (!re) throw new Error("internal: missing reduction");
			const reIsInt = kernel.exp.dtype === "int32" || kernel.exp.dtype === "uint32";
			const vecAccs = gidxs.map(() => cg.local.declare(reIsInt ? cg.i32x4 : cg.f32x4));
			const initializeIdentityAccumulators = () => {
				for (const acc of vecAccs) {
					if (reIsInt) {
						cg.i32.const(re.identity);
						cg.i32x4.splat();
					} else {
						cg.f32.const(re.identity);
						cg.f32x4.splat();
					}
					cg.local.set(acc);
				}
			};
			const loadPartialAccumulators = () => {
				for (let i = 0; i < gidxs.length; i++) {
					emitOutputAddress(gidxs[i]);
					if (reIsInt) cg.i32x4.load(4);
					else cg.f32x4.load(4);
					cg.local.set(vecAccs[i]);
				}
			};
			if (ridxStart === void 0) initializeIdentityAccumulators();
			else {
				cg.local.get(ridxStart);
				cg.i32.eqz();
				cg.if(cg.void);
				initializeIdentityAccumulators();
				cg.else();
				loadPartialAccumulators();
				cg.end();
			}
			const ridx = cg.local.declare(cg.i32);
			const emitReductionStep = () => {
				const pointerValueCache = /* @__PURE__ */ new Map();
				for (let i = 0; i < gidxs.length; i++) {
					emitSimdReductionOp(cg, re, reIsInt, emitSimdExpWithAccumulator({
						gidx: gidxs[i],
						ridx
					}, pointerMaps[i], pointerValueCache, vecAccs[i]));
					cg.local.set(vecAccs[i]);
				}
				incrementReductionPointers(cg, uniquePointers);
			};
			if (ridxStart === void 0) setLocalConst(ridx, 0);
			else copyLocal(ridx, ridxStart);
			const emitReductionLoopBody = () => {
				emitReductionStep();
				bumpLocal(ridx, 1);
			};
			if (ridxEnd === void 0) emitLoopWhileConstLt(ridx, re.size, emitReductionLoopBody);
			else emitLoopWhileLocalLt(ridx, ridxEnd, emitReductionLoopBody);
			const storeRawAccumulators = () => {
				for (let i = 0; i < gidxs.length; i++) {
					emitOutputAddress(gidxs[i]);
					cg.local.get(vecAccs[i]);
					cg.v128.store(4);
				}
			};
			if (hasIdentityEpilogue) {
				storeRawAccumulators();
				return;
			}
			const storeEpilogueAccumulators = () => {
				const laneGidx = cg.local.declare(cg.i32);
				const laneAcc = cg.local.declare(reIsInt ? cg.i32 : cg.f32);
				for (let i = 0; i < gidxs.length; i++) for (let lane = 0; lane < 4; lane++) {
					cg.local.get(kernel.nargs);
					cg.local.get(gidxs[i]);
					if (lane > 0) {
						cg.i32.const(lane);
						cg.i32.add();
					}
					cg.local.tee(laneGidx);
					cg.i32.const(byteWidth(kernel.dtype));
					cg.i32.mul();
					cg.i32.add();
					cg.local.get(vecAccs[i]);
					if (reIsInt) cg.i32x4.extract_lane(lane);
					else cg.f32x4.extract_lane(lane);
					cg.local.set(laneAcc);
					translateExp(cg, funcs, tune.epilogue, {
						acc: laneAcc,
						gidx: laneGidx
					});
					dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
				}
			};
			if (ridxEnd !== void 0) {
				cg.local.get(ridxEnd);
				cg.i32.const(re.size);
				cg.i32.lt_u();
				cg.if(cg.void);
				storeRawAccumulators();
				cg.else();
				storeEpilogueAccumulators();
				cg.end();
			} else storeEpilogueAccumulators();
		};
		const initializePointerMaps = (groups, candidates, keyFor, ctxFor, ridxOffset) => {
			const pointerMaps = groups.map(() => /* @__PURE__ */ new Map());
			const sharedPointers = /* @__PURE__ */ new Map();
			const uniquePointers = [];
			for (let i = 0; i < groups.length; i++) {
				const group = groups[i];
				for (const candidate of candidates) {
					const key = keyFor(candidate, group, i);
					let plan = sharedPointers.get(key);
					if (!plan) {
						plan = initializeReductionPointer(cg, funcs, candidate, ctxFor(group), key, ridxOffset);
						sharedPointers.set(key, plan);
						uniquePointers.push(plan);
					}
					pointerMaps[i].set(candidate.exp, plan);
				}
			}
			return {
				pointerMaps,
				uniquePointers
			};
		};
		const emitElementwiseSimdStep = () => {
			emitOutputAddress(gidx);
			translateExpSimd(cg, funcs, tune.exp, { gidx }, expStrides);
			cg.v128.store(4);
			bumpLocal(gidx, 4);
		};
		const emitKSimdReductionForGroups = (groups, pointerMaps, uniquePointers) => {
			if (!re) throw new Error("internal: missing reduction");
			if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");
			const vecAccs = groups.map(() => cg.local.declare(cg.f32x4));
			for (const acc of vecAccs) {
				cg.f32.const(re.identity);
				cg.f32x4.splat();
				cg.local.set(acc);
			}
			const ridx = cg.local.declare(cg.i32);
			setLocalConst(ridx, 0);
			emitLoopWhileConstLt(ridx, re.size, () => {
				for (let u = 0; u < kSimdTilePlan.kUnroll; u++) {
					const pointerValueCache = /* @__PURE__ */ new Map();
					for (let i = 0; i < groups.length; i++) {
						if (!emitSimdExpWithAccumulator({
							gidx: groups[i].gidx,
							ridx
						}, pointerMaps[i], pointerValueCache, vecAccs[i])) cg.f32x4.add();
						cg.local.set(vecAccs[i]);
					}
					incrementReductionPointers(cg, uniquePointers, 4);
				}
				bumpLocal(ridx, 4 * kSimdTilePlan.kUnroll);
			});
			for (let i = 0; i < groups.length; i++) {
				const acc = cg.local.declare(cg.f32);
				for (let lane = 0; lane < 4; lane++) {
					cg.local.get(vecAccs[i]);
					cg.f32x4.extract_lane(lane);
					if (lane > 0) cg.f32.add();
				}
				cg.local.set(acc);
				emitOutputAddress(groups[i].gidx);
				translateExp(cg, funcs, tune.epilogue, {
					acc,
					gidx: groups[i].gidx
				});
				dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
			}
		};
		const emitKSimdReductionLoop = (plan) => {
			const colTile = cg.local.declare(cg.i32);
			const col = cg.local.declare(cg.i32);
			const rowTileBase = cg.local.declare(cg.i32);
			const rowOffset = cg.local.declare(cg.i32);
			const rowBase = cg.local.declare(cg.i32);
			const emitBlock = () => {
				const groups = [];
				for (let row = 0; row < plan.microRows; row++) for (let colOffset = 0; colOffset < plan.microCols; colOffset++) groups.push({
					gidx: declareTileGidx(rowBase, col, row * plan.tileSize, colOffset),
					row,
					col: colOffset
				});
				if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");
				const { pointerMaps, uniquePointers } = initializePointerMaps(groups, kSimdReductionPointerCandidates, (candidate, group, i) => kReductionPointerShareKey(candidate, expStrides.get(candidate.exp) ?? { kind: "gather" }, kSimdTilePlan.tileSize, group.row, group.col, i), (group) => ({ gidx: group.gidx }));
				emitKSimdReductionForGroups(groups, pointerMaps, uniquePointers);
			};
			emitLoopWhileLocalLt(gidx, paramEnd, () => {
				copyLocal(rowTileBase, gidx);
				setLocalConst(colTile, 0);
				emitLoopWhileConstLt(colTile, plan.tileSize, () => {
					setLocalConst(rowOffset, 0);
					emitRowTileLoop(rowOffset, rowTileBase, plan.tileRows, plan.tileSize, () => {
						copyLocal(col, colTile);
						emitLoopWhileLtAndConstLt(col, () => emitLocalPlusConst(colTile, plan.tileCols), plan.tileSize, () => {
							setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
							emitBlock();
							bumpLocal(col, plan.microCols);
						});
						bumpLocal(rowOffset, plan.microRows);
					});
					bumpLocal(colTile, plan.tileCols);
				});
				bumpLocal(gidx, plan.tileRows * plan.tileSize);
			});
		};
		const emitTiledSimdReductionLoop = (plan) => {
			const colTile = cg.local.declare(cg.i32);
			const col = cg.local.declare(cg.i32);
			const kTile = cg.local.declare(cg.i32);
			const tileEnd = cg.local.declare(cg.i32);
			const rowTileBase = cg.local.declare(cg.i32);
			const rowOffset = cg.local.declare(cg.i32);
			const rowBase = cg.local.declare(cg.i32);
			const emitRowBlock = (microVectors) => {
				const groups = [];
				for (let row = 0; row < plan.microRows; row++) for (let vector = 0; vector < microVectors; vector++) groups.push({
					gidx: declareTileGidx(rowBase, col, row * plan.tileSize, vector * 4),
					row,
					vector
				});
				const { pointerMaps, uniquePointers } = initializePointerMaps(groups, simdReductionPointerCandidates, (candidate, group, i) => pointerShareKey(candidate, group.row, group.vector, i), (group) => ({ gidx: group.gidx }), kTile);
				emitSimdReductionForGidxs(groups.map((group) => group.gidx), pointerMaps, uniquePointers, kTile, tileEnd);
			};
			emitLoopWhileLocalLt(gidx, paramEnd, () => {
				copyLocal(rowTileBase, gidx);
				setLocalConst(colTile, 0);
				emitLoopWhileConstLt(colTile, plan.tileSize, () => {
					setLocalConst(kTile, 0);
					emitLoopWhileConstLt(kTile, re.size, () => {
						emitLocalPlusConst(kTile, plan.tileK);
						cg.local.set(tileEnd);
						setLocalConst(rowOffset, 0);
						emitRowTileLoop(rowOffset, rowTileBase, plan.tileRows, plan.tileSize, () => {
							copyLocal(col, colTile);
							emitLoopWhileBlockFits(col, plan.microVectors * 4, () => emitLocalPlusConst(colTile, plan.tileVectors * 4), plan.tileSize, () => {
								setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
								emitRowBlock(plan.microVectors);
								bumpLocal(col, plan.microVectors * 4);
							});
							emitLoopWhileLtAndConstLt(col, () => emitLocalPlusConst(colTile, plan.tileVectors * 4), plan.tileSize, () => {
								setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
								emitRowBlock(1);
								bumpLocal(col, 4);
							});
							bumpLocal(rowOffset, plan.microRows);
						});
						bumpLocal(kTile, plan.tileK);
					});
					bumpLocal(colTile, plan.tileVectors * 4);
				});
				bumpLocal(gidx, plan.tileRows * plan.tileSize);
			});
		};
		const emitGuardedFastPath = (alignment, emit) => {
			emitAlignmentGuard(cg, paramBegin, paramEnd, alignment);
			emit();
			cg.return();
			cg.end();
		};
		if (kSimdTilePlan) emitGuardedFastPath(kSimdTilePlan.microRows * kSimdTilePlan.tileSize, () => emitKSimdReductionLoop(kSimdTilePlan));
		if (useSimd) {
			if (simdTilePlan) emitGuardedFastPath(simdTilePlan.microRows * simdTilePlan.tileSize, () => emitTiledSimdReductionLoop(simdTilePlan));
			if (!re) emitGuardedFastPath(4, () => {
				emitLoopWhileLocalLt(gidx, paramEnd, emitElementwiseSimdStep);
			});
		}
		emitLoopWhileLocalLt(gidx, paramEnd, () => {
			emitOutputAddress(gidx);
			if (re) {
				const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
				dty(cg, null, kernel.exp.dtype).const(re.identity);
				cg.local.set(acc);
				const ridx = cg.local.declare(cg.i32);
				const emitReductionStep = () => {
					translateExp(cg, funcs, tune.exp, {
						gidx,
						ridx
					});
					emitScalarReductionOp(cg, re, acc);
					cg.local.set(acc);
				};
				setLocalConst(ridx, 0);
				emitLoopWhileConstLt(ridx, re.size, () => {
					emitReductionStep();
					bumpLocal(ridx, 1);
				});
				translateExp(cg, funcs, tune.epilogue, {
					acc,
					gidx
				});
			} else translateExp(cg, funcs, tune.exp, { gidx });
			dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
			bumpLocal(gidx, 1);
		});
	});
	cg.export(kernelFunc, "kernel");
	const tiledPlan = simdTilePlan ?? kSimdTilePlan;
	return {
		bytes: cg.finish(),
		workSize: kernel.size,
		chunkAlignment: tiledPlan ? tiledPlan.tileRows * tiledPlan.tileSize : 16,
		minWorkPerWorker: tiledPlan && kernel.size / tiledPlan.tileSize >= 1024 ? tiledPlan.tileSize * 32 : void 0
	};
}
//#endregion
//#region src/backend/wasm.ts
const compiledProgramCache = /* @__PURE__ */ new Map();
/** Backend that compiles into WebAssembly bytecode for immediate execution. */
var WasmBackend = class {
	type = "wasm";
	maxArgs = 64;
	#memory;
	#nextSlot;
	#allocator;
	#buffers;
	#workerPool;
	#pendingWork = /* @__PURE__ */ new Map();
	constructor() {
		this.#memory = hasSharedArrayBuffer() ? new WebAssembly.Memory({
			initial: 0,
			maximum: 65536,
			shared: true
		}) : new WebAssembly.Memory({ initial: 0 });
		this.#allocator = new WasmAllocator(this.#memory);
		this.#nextSlot = 1;
		this.#buffers = /* @__PURE__ */ new Map();
		this.#workerPool = createWorkerPool(this.#memory);
	}
	malloc(size, initialData) {
		const ptr = this.#allocator.malloc(size);
		if (initialData) {
			if (initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
			new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
		}
		const slot = this.#nextSlot++;
		this.#buffers.set(slot, {
			ptr,
			size,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) {
			this.#allocator.free(buffer.ptr);
			this.#buffers.delete(slot);
			this.#pendingWork.delete(slot);
		}
	}
	async read(slot, start, count) {
		const epoch = this.#pendingWork.get(slot);
		if (epoch) await this.#workerPool.waitForEpoch(epoch);
		return this.#readData(slot, start, count);
	}
	readSync(slot, start, count) {
		const epoch = this.#pendingWork.get(slot);
		if (epoch && this.#workerPool.epoch < epoch) throw new Error("cannot read synchronously from a slot with async work");
		return this.#readData(slot, start, count);
	}
	#readData(slot, start, count) {
		const buffer = this.#getBuffer(slot);
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.byteLength - start;
		if (hasSharedArrayBuffer() && buffer.buffer instanceof SharedArrayBuffer) return new Uint8Array(buffer.slice(start, start + count));
		else return buffer.slice(start, start + count);
	}
	async prepareKernel(kernel) {
		const kernelHash = FpHash.hash(kernel).toString();
		return new Executable(kernel, {
			program: await runWithCacheAsync(compiledProgramCache, kernelHash, async () => {
				const { bytes, ...metadata } = codegenWasm(kernel);
				return {
					kernelHash,
					module: await WebAssembly.compile(bytes),
					...metadata
				};
			}),
			sync: false
		});
	}
	prepareKernelSync(kernel) {
		const kernelHash = FpHash.hash(kernel).toString();
		return new Executable(kernel, {
			program: runWithCache(compiledProgramCache, kernelHash, () => {
				const { bytes, ...metadata } = codegenWasm(kernel);
				return {
					kernelHash,
					module: new WebAssembly.Module(bytes),
					...metadata
				};
			}),
			sync: true
		});
	}
	async prepareRoutine(routine) {
		return new Executable(routine, {
			program: void 0,
			sync: false
		});
	}
	prepareRoutineSync(routine) {
		return new Executable(routine, {
			program: void 0,
			sync: true
		});
	}
	dispatch(exe, inputs, outputs) {
		const tracing = isTracing();
		const start = tracing ? performance.now() : 0;
		const { sync } = exe.data;
		if (exe.source instanceof Routine) if (this.#workerPool && !sync) {
			const routine = {
				name: exe.source.name,
				type: exe.source.type,
				params: exe.source.params
			};
			const retainedSlots = [...inputs, ...outputs];
			for (const slot of retainedSlots) this.incRef(slot);
			const epoch = this.#workerPool.dispatchRoutine(routine, inputs.map((slot) => {
				const { ptr, size } = this.#buffers.get(slot);
				return {
					ptr,
					size
				};
			}), outputs.map((slot) => {
				const { ptr, size } = this.#buffers.get(slot);
				return {
					ptr,
					size
				};
			}));
			for (const slot of outputs) this.#pendingWork.set(slot, epoch);
			this.#workerPool.waitForEpoch(epoch).then(() => {
				for (const slot of outputs) if (this.#pendingWork.get(slot) === epoch) this.#pendingWork.delete(slot);
				for (const slot of retainedSlots) this.decRef(slot);
			});
		} else {
			if (inputs.some((slot) => {
				const epoch = this.#pendingWork.get(slot);
				return epoch && this.#workerPool.epoch < epoch;
			})) throw new Error("cannot dispatch wasm routine synchronously with pending async work");
			runCpuRoutine(exe.source, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
		}
		else {
			const { program } = exe.data;
			const ptrs = [...inputs, ...outputs].map((slot) => this.#buffers.get(slot).ptr);
			if (this.#workerPool && !sync) {
				const retainedSlots = [...inputs, ...outputs];
				for (const slot of retainedSlots) this.incRef(slot);
				const epoch = this.#workerPool.dispatch(program.kernelHash, program.module, ptrs, program.workSize, program.chunkAlignment, program.minWorkPerWorker);
				for (const slot of outputs) this.#pendingWork.set(slot, epoch);
				this.#workerPool.waitForEpoch(epoch).then(() => {
					for (const slot of outputs) if (this.#pendingWork.get(slot) === epoch) this.#pendingWork.delete(slot);
					for (const slot of retainedSlots) this.decRef(slot);
				});
			} else {
				if (inputs.some((slot) => {
					const epoch = this.#pendingWork.get(slot);
					return epoch && this.#workerPool.epoch < epoch;
				})) throw new Error("cannot dispatch synchronously with pending async work");
				const func = new WebAssembly.Instance(program.module, { env: { memory: this.#memory } }).exports.kernel;
				func(...ptrs, 0, program.workSize);
			}
		}
		if (tracing) emitTrace("wasm", traceSourceInfo(exe.source), start, performance.now());
	}
	#getBuffer(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
	}
};
//#endregion
//#region src/backend.ts
const devices = [
	"cpu",
	"wasm",
	"webgpu",
	"webgl"
];
const initializedBackends = /* @__PURE__ */ new Map();
initializedBackends.set("cpu", new CpuBackend());
if (typeof WebAssembly !== "undefined") initializedBackends.set("wasm", new WasmBackend());
let defaultBackend = initializedBackends.has("wasm") ? "wasm" : "cpu";
/** Configure the default device for arrays. */
function defaultDevice(device) {
	if (device !== void 0) if (initializedBackends.has(device)) defaultBackend = device;
	else throw new Error(`Backend not initialized: ${device}`);
	return defaultBackend;
}
/**
* Initialize `jax-js` library backends.
*
* By default, this will initialize all available backends. If one or more
* backends is provided, only attempt to initialize those. Returns a list of
* available backends.
*/
async function init(...devicesToInit) {
	if (devicesToInit.length === 0) devicesToInit = devices;
	const promises = [];
	for (const device of new Set(devicesToInit)) if (!initializedBackends.has(device)) promises.push((async () => {
		const backend = await createBackend(device);
		if (backend) initializedBackends.set(device, backend);
	})());
	await Promise.all(promises);
	return Array.from(initializedBackends.keys());
}
/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(device) {
	if (device === "cpu") return new CpuBackend();
	else if (device === "wasm") {
		if (typeof WebAssembly === "undefined") return null;
		return new WasmBackend();
	} else if (device === "webgpu") {
		if (!navigator.gpu) return null;
		const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) return null;
		const { WebGPUBackend } = await import("./webgpu-D5edCWOx.js");
		const importantLimits = [
			"maxBufferSize",
			"maxComputeInvocationsPerWorkgroup",
			"maxComputeWorkgroupSizeX",
			"maxComputeWorkgroupSizeY",
			"maxComputeWorkgroupSizeZ",
			"maxComputeWorkgroupStorageSize",
			"maxComputeWorkgroupsPerDimension",
			"maxStorageBufferBindingSize",
			"maxStorageBuffersPerShaderStage",
			"maxStorageTexturesPerShaderStage"
		];
		const requestedFeatures = ["shader-f16", "timestamp-query"];
		try {
			return new WebGPUBackend(await adapter.requestDevice({
				requiredLimits: Object.fromEntries(importantLimits.map((limit) => [limit, adapter.limits[limit]])),
				requiredFeatures: requestedFeatures.filter((feature) => adapter.features.has(feature))
			}));
		} catch (error) {
			console.error("Unexpected error requesting WebGPU device:", error);
			return null;
		}
	} else if (device === "webgl") {
		if (typeof WebGL2RenderingContext === "undefined") return null;
		const gl = new OffscreenCanvas(0, 0).getContext("webgl2", {
			alpha: false,
			antialias: false,
			premultipliedAlpha: false,
			preserveDrawingBuffer: false,
			depth: false,
			stencil: false,
			failIfMajorPerformanceCaveat: true
		});
		if (!gl) return null;
		if (!gl.getExtension("EXT_color_buffer_float")) return null;
		const { WebGLBackend } = await import("./webgl-d1MbwhuX.js");
		return new WebGLBackend(gl);
	} else throw new Error(`Backend not found: ${device}`);
}
/** Retrieve a backend that has been initialized. */
function getBackend(device) {
	device = device ?? defaultBackend;
	const backend = initializedBackends.get(device);
	if (!backend) throw new Error(`${device} backend not ready, call init() first`);
	return backend;
}
var Executable = class {
	source;
	data;
	constructor(source, data) {
		this.source = source;
		this.data = data;
	}
};
var SlotError = class extends Error {
	constructor(slot) {
		super(`Used a buffer that is invalid or already freed: ${slot}`);
	}
};
var UnsupportedOpError = class extends Error {
	constructor(op, dtype, device, arg) {
		let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
		if (arg !== void 0) msg += ` with arg ${JSON.stringify(arg)}`;
		super(msg);
	}
};
var UnsupportedRoutineError = class extends Error {
	constructor(name, device) {
		super(`routine '${name}' is not supported in ${device} backend`);
	}
};
/**
* If the WebGPU backend has been initialized, return the `GPUDevice` that this
* backend runs on. This is useful for sharing buffers.
*/
function getWebGPUDevice() {
	const backend = initializedBackends.get("webgpu");
	if (!backend) throw new Error("WebGPU backend not initialized, call init('webgpu') first");
	return backend.device;
}
//#endregion
export { range as $, accessorAluExp as A, checkAxis as B, ScatterOp as C, DType as D, AluVar as E, isFloatDtype as F, generalBroadcast as G, checkSquare as H, promoteTypes as I, isPermutation as J, invertPermutation as K, DEBUG as L, byteWidth as M, dtypedArray as N, Kernel as O, dtypedJsArray as P, prod as Q, FpHash as R, Routines as S, AluGroup as T, deepEqual as U, checkInts as V, findPow2 as W, normalizeAxis as X, mapSetUnion as Y, partitionList as Z, tuneWebgpu as _, defaultDevice as a, toposort as at, unravelAlu as b, getWebGPUDevice as c, zipn as ct, isTracing as d, recursiveFlatten as et, onFlushTrace as f, tuneNullopt as g, traceSourceInfo as h, UnsupportedRoutineError as i, strip1 as it, accessorGlobal as j, Reduction as k, init as l, PPrint as lt, stopTrace as m, SlotError as n, runWithCache as nt, devices as o, unzip2 as ot, startTrace as p, isNumberPair as q, UnsupportedOpError as r, setDebug as rt, getBackend as s, zip as st, Executable as t, rep as tt, emitTrace as u, ShapeTracker as v, AluExp as w, Routine as x, View as y, assertNonNull as z };
