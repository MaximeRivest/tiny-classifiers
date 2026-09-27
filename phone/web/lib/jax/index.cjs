Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
//#region \0rolldown/runtime.js
var __defProp = Object.defineProperty;
var __exportAll = (all, no_symbols) => {
	let target = {};
	for (var name in all) __defProp(target, name, {
		get: all[name],
		enumerable: true
	});
	if (!no_symbols) __defProp(target, Symbol.toStringTag, { value: "Module" });
	return target;
};
//#endregion
const require_backend = require("./backend-TLHPgbCA.cjs");
//#region src/frontend/convolution.ts
/**
* Check that the shapes and parameters passed to convolution are valid.
* Expected shapes of the lhs and rhs of the convolution are:
*
* - `lhsShape = [*vmapDims, batchSize, inChannels, spatialDims...]`
* - `rhsShape = [*vmapDims, outChannels, inChannels, kernelSize...]`
*
* If the check succeeds, returns the output shape.
*/
function checkConvShape(lhsShape, rhsShape, { vmapDims, strides, padding, lhsDilation, rhsDilation }) {
	if (lhsShape.length !== rhsShape.length) throw new Error(`conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`);
	const n = lhsShape.length - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	if (strides.length !== n) throw new Error("conv() strides != spatial dims");
	if (padding.length !== n) throw new Error("conv() padding != spatial dims");
	if (lhsDilation.length !== n) throw new Error("conv() lhsDilation != spatial dimensions");
	if (rhsDilation.length !== n) throw new Error("conv() rhsDilation != spatial dimensions");
	if (lhsShape[vmapDims + 1] !== rhsShape[vmapDims + 1]) throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
	const outShape = [
		...require_backend.generalBroadcast(lhsShape.slice(0, vmapDims), rhsShape.slice(0, vmapDims)),
		lhsShape[vmapDims],
		rhsShape[vmapDims]
	];
	for (let i = 0; i < n; i++) {
		if (strides[i] <= 0 || !Number.isInteger(strides[i])) throw new Error(`conv() strides[${i}] must be a positive integer`);
		if (padding[i].length !== 2 || !padding[i].every(Number.isInteger)) throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
		if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i])) throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
		if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i])) throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);
		const [x, k] = [lhsShape[i + vmapDims + 2], rhsShape[i + vmapDims + 2]];
		if (k <= 0) throw new Error("conv() kernel size must be positive");
		const [pl, pr] = padding[i];
		if (pl < -x || pr < -x || pl + pr < -x) throw new Error(`conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`);
		const kernelSize = (k - 1) * rhsDilation[i] + 1;
		const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
		if (kernelSize > inSize) throw new Error(`conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`);
		outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
	}
	return outShape;
}
function checkPoolShape(inShape, window, strides) {
	if (strides.length !== window.length) throw new Error("pool() strides != window dims");
	if (window.length > inShape.length) throw new Error("pool() window has more dimensions than input");
	const outShape = inShape.slice(0, inShape.length - window.length);
	for (let i = 0; i < window.length; i++) {
		const k = window[i];
		const s = strides[i];
		const size = inShape[inShape.length - window.length + i];
		if (k <= 0 || !Number.isInteger(k)) throw new Error(`pool() window[${i}] must be a positive integer`);
		if (k > size) throw new Error(`pool() window[${i}]=${k} > input size ${size}`);
		if (s <= 0 || !Number.isInteger(s)) throw new Error(`pool() strides[${i}] must be a positive integer`);
		outShape.push(Math.ceil((size - k + 1) / s));
	}
	return outShape.concat(window);
}
/**
* Takes a shape tracker and a kernel size `ks`, then reshapes it so the last
* `ks.length` dimensions become `2 * ks.length` dimensions by treating them as
* spatial dimensions convolved with a kernel.
*
* The resulting array can be multiplied with a kernel of shape `ks`, then
* reduced along the last `ks.length` dimensions for a convolution.
*
* Reference: https://github.com/tinygrad/tinygrad/blob/v0.10.3/tinygrad/tensor.py#L2097
*/
function pool(st, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (st.shape.length < ks.length) throw new Error("pool() called with too many dimensions");
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	if (strides.some((s) => s <= 0 || !Number.isInteger(s))) throw new Error("pool() strides must be positive integers");
	if (dilation.some((d) => d <= 0 || !Number.isInteger(d))) throw new Error("pool() dilation must be positive integers");
	const noop = st.shape.slice(0, -ks.length);
	const i_ = st.shape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, o * s - i])]);
		st = st.reshape([...noop, ...require_backend.zip(o_, s_).flatMap(([o, s]) => [o, s])]).shrink([...noop.map((x) => [0, x]), ...require_backend.zip(o_, ks).flatMap(([o, k]) => [[0, o], [0, k]])]);
		st = st.permute([
			...require_backend.range(noop.length),
			...ks.map((_, j) => noop.length + 2 * j),
			...ks.map((_, j) => noop.length + 2 * j + 1)
		]);
		return st;
	}
	const kidf = require_backend.zipn(ks, i_, d_, require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1))));
	st = st.repeat([...require_backend.rep(noop.length, 1), ...kidf.map(([k, i, d, f]) => Math.ceil(k * (i * f + d) / i))]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kidf.map(([k, i, d, f]) => [0, k * (i * f + d)])]).reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o, s]) => [[0, k], [0, o * s]])]).reshape([...noop, ...kos.flat(1)]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o]) => [
		[0, k],
		[0, o],
		[0, 1]
	])]).reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);
	st = st.permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/**
* Perform the transpose of pool, directly undo-ing a pool() operation.
*
* Note that since pool repeats the input, the transpose operation technically
* should include a sum reduction. This function doesn't perform the reduction,
* which should be done on the last `k` axes of the returned shape.
*/
function poolTranspose(st, inShape, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	const noop = inShape.slice(0, -ks.length);
	const i_ = inShape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j])]);
		st = st.pad([...noop.map(() => [0, 0]), ...require_backend.zip(s_, ks).flatMap(([s, k]) => [[0, 0], [0, s - k]])]).reshape([...noop, ...require_backend.zip(o_, s_).map(([o, s]) => o * s)]);
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, i - o * s])]);
		return st.reshape(st.shape.concat(require_backend.rep(ks.length, 1)));
	}
	if (!require_backend.deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length))) throw new Error("poolTranspose() called with mismatched output shape");
	const kidf = require_backend.zipn(ks, i_, d_, require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1))));
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [
		k,
		o,
		1
	])]).pad([...noop.map(() => [0, 0]), ...s_.flatMap((s) => [
		[0, 0],
		[0, 0],
		[0, s - 1]
	])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([...noop.map(() => [0, 0]), ...kidf.flatMap(([_k, i, d, f], j) => [[0, 0], [0, i * f + d - o_[j] * s_[j]]])]);
	st = st.reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))]).pad([...noop.map(() => [0, 0]), ...kidf.map(([k, i, d, f]) => [0, Math.ceil(k * (i * f + d) / i) * i - k * (i * f + d)])]);
	st = st.reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [Math.ceil(k * (i * f + d) / i), i])]).permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/** Applies dilation to an array directly, for transposed convolution. */
function applyDilation(st, dilation) {
	if (dilation.every((s) => s === 1)) return st;
	const s_ = dilation;
	const n = s_.length;
	const prefix = st.shape.slice(0, -n);
	const k_ = st.shape.slice(-n);
	st = st.reshape([...prefix, ...k_.flatMap((k) => [k, 1])]);
	st = st.pad([...prefix.map(() => [0, 0]), ...s_.flatMap((s) => [[0, 0], [0, s - 1]])]);
	st = st.reshape([...prefix, ...k_.map((k, i) => k * s_[i])]);
	st = st.shrink([...prefix.map((p) => [0, p]), ...k_.map((k, i) => [0, (k - 1) * s_[i] + 1])]);
	return st;
}
/**
* Prepare for a convolution between two arrays.
*
* This does not check the validity of the shapes, which should be checked
* beforehand using `checkConvShape()`.
*/
function prepareConv(stX, stY, params) {
	const v = params.vmapDims;
	const n = stX.shape.length - 2 - v;
	const vmapShape = stX.shape.slice(0, v);
	stX = applyDilation(stX, params.lhsDilation);
	const ks = stY.shape.slice(v + 2);
	stX = stX.padOrShrink([...require_backend.rep(v + 2, [0, 0]), ...params.padding]);
	stX = pool(stX, ks, params.strides, params.rhsDilation);
	stX = stX.moveaxis(v + 1, v + n + 1).reshape([
		...vmapShape,
		stX.shape[v],
		1,
		...stX.shape.slice(v + 2, v + n + 2),
		stX.shape[v + 1] * require_backend.prod(ks)
	]);
	stY = stY.reshape([
		...vmapShape,
		1,
		stY.shape[v],
		...require_backend.rep(n, 1),
		stY.shape[v + 1] * require_backend.prod(ks)
	]);
	return [stX, stY];
}
//#endregion
//#region src/tree.ts
var tree_exports = /* @__PURE__ */ __exportAll({
	JsTreeDef: () => JsTreeDef,
	NodeType: () => NodeType,
	dispose: () => dispose,
	flatten: () => flatten,
	leaves: () => leaves,
	map: () => map,
	ref: () => ref,
	structure: () => structure,
	unflatten: () => unflatten
});
const JsArray$4 = globalThis.Array;
let NodeType = /* @__PURE__ */ function(NodeType) {
	NodeType["Array"] = "Array";
	NodeType["Object"] = "Object";
	NodeType["Leaf"] = "Leaf";
	return NodeType;
}({});
/** Represents the structure of a JsTree. */
var JsTreeDef = class JsTreeDef {
	nodeType;
	nodeMetadata;
	childTreedefs;
	static leaf = new JsTreeDef("Leaf", null, []);
	constructor(nodeType, nodeMetadata, childTreedefs) {
		this.nodeType = nodeType;
		this.nodeMetadata = nodeMetadata;
		this.childTreedefs = childTreedefs;
	}
	/** Get the total number of leaves in the tree. */
	get size() {
		return this.nodeType === "Leaf" ? 1 : this.childTreedefs.reduce((a, b) => a + b.size, 0);
	}
	/** Returns a string representation of this tree definition. */
	toString(root = true) {
		if (root) return "JsTreeDef(" + this.toString(false) + ")";
		switch (this.nodeType) {
			case "Leaf": return "*";
			case "Array": return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
			case "Object": {
				const parts = [];
				for (let i = 0; i < this.childTreedefs.length; i++) parts.push(`${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`);
				return `{${parts.join(", ")}}`;
			}
		}
	}
	/** Compare this tree definition with another. */
	equals(other) {
		return this.nodeType === other.nodeType && require_backend.deepEqual(this.nodeMetadata, other.nodeMetadata) && this.childTreedefs.length === other.childTreedefs.length && this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]));
	}
};
function quoteObjectKey(key) {
	if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)) return key;
	return JSON.stringify(key);
}
/** Flatten a structured object, returning the tree definition. */
function flatten(tree) {
	const leaves = [];
	return [leaves, _flatten(tree, leaves)];
}
function _flatten(tree, leaves) {
	if (JsArray$4.isArray(tree)) return new JsTreeDef("Array", null, tree.map((c) => _flatten(c, leaves)));
	else if (typeof tree === "object" && tree !== null && tree.constructor === Object) {
		const [keys, values] = require_backend.unzip2(Object.entries(tree));
		return new JsTreeDef("Object", keys, values.map((c) => _flatten(c, leaves)));
	} else {
		leaves.push(tree);
		return JsTreeDef.leaf;
	}
}
/** Get the leaves of a tree. */
function leaves(tree) {
	return flatten(tree)[0];
}
/** Get the treedef for a tree. */
function structure(tree) {
	return flatten(tree)[1];
}
/** Reconstruct a structured object from the flattened representation. */
function unflatten(treedef, leaves) {
	return _unflatten(treedef, leaves[Symbol.iterator]());
}
function _unflatten(treedef, leaves) {
	switch (treedef.nodeType) {
		case "Leaf": {
			const { value, done } = leaves.next();
			if (done) throw new TypeError("Ran out of leaves while unflattening JsTree");
			return value;
		}
		case "Array": return treedef.childTreedefs.map((c) => _unflatten(c, leaves));
		case "Object": {
			const obj = {};
			for (let i = 0; i < treedef.childTreedefs.length; i++) obj[treedef.nodeMetadata[i]] = _unflatten(treedef.childTreedefs[i], leaves);
			return obj;
		}
	}
}
/** Maps a multi-input function over pytree args to produce a new pytree. */
function map(fn, tree, ...rest) {
	const [leaves, treedef] = flatten(tree);
	const restLeaves = rest.map((x) => flatten(x)[0]);
	const resultLeaves = [];
	for (let i = 0; i < leaves.length; i++) resultLeaves.push(fn(leaves[i], ...restLeaves.map((x) => x[i])));
	return unflatten(treedef, resultLeaves);
}
/** Take a reference of every array in a tree. */
function ref(tree) {
	return map((x) => x instanceof Tracer ? x.ref : x, tree);
}
/** Dispose every array in a tree. */
function dispose(tree) {
	if (tree) map((x) => x instanceof Tracer ? x.dispose() : void 0, tree);
}
//#endregion
//#region src/frontend/core.ts
/** @file Core library internals and interpreter stack, based on Autodidax. */
const routinePrimitives = /* @__PURE__ */ new Map([
	["sort", "Sort"],
	["argsort", "Argsort"],
	["scatter", "Scatter"],
	["triangular_solve", "TriangularSolve"],
	["cholesky", "Cholesky"],
	["lu", "LU"],
	["jacobi_eigh", "JacobiEigh"],
	["fft", "Fft"]
]);
function add$1(x, y) {
	return bind1("add", [x, y]);
}
function mul(x, y) {
	return bind1("mul", [x, y]);
}
function idiv(x, y) {
	return bind1("idiv", [x, y]);
}
function mod(x, y) {
	return bind1("mod", [x, y]);
}
function min$1(x, y) {
	return bind1("min", [x, y]);
}
function max$1(x, y) {
	return bind1("max", [x, y]);
}
function bitCombine(x, y, op) {
	return bind1("bit_combine", [x, y], { op });
}
function bitShift(x, y, op) {
	return bind1("bit_shift", [x, y], { op });
}
function bitCount(x) {
	return bind1("bit_count", [x]);
}
function neg(x) {
	return bind1("neg", [x]);
}
function reciprocal$1(x) {
	return bind1("reciprocal", [x]);
}
function floor$1(x) {
	return bind1("floor", [x]);
}
function ceil$1(x) {
	return bind1("ceil", [x]);
}
function stopGradient$1(x) {
	return bind1("stop_gradient", [x]);
}
function cast(x, dtype) {
	return bind1("cast", [x], { dtype });
}
function bitcast(x, dtype) {
	return bind1("bitcast", [x], { dtype });
}
function sin$1(x) {
	return bind1("sin", [x]);
}
function cos$1(x) {
	return bind1("cos", [x]);
}
function asin$1(x) {
	return bind1("asin", [x]);
}
function atan$1(x) {
	return bind1("atan", [x]);
}
function exp$1(x) {
	return bind1("exp", [x]);
}
function log$1(x) {
	return bind1("log", [x]);
}
function erf$1(x) {
	return bind1("erf", [x]);
}
function erfc$1(x) {
	return bind1("erfc", [x]);
}
function sqrt$1(x) {
	return bind1("sqrt", [x]);
}
function reduce(x, op, axis = null, opts) {
	if (!require_backend.AluGroup.Reduce.has(op)) throw new TypeError(`Invalid reduce operation: ${op}`);
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	const originalShape = getShape(x);
	let result = bind1("reduce", [x], {
		op,
		axis
	});
	if (opts?.keepdims) result = result.reshape(originalShape.map((dim, i) => axis.includes(i) ? 1 : dim));
	return result;
}
function dot$2(x, y) {
	return bind1("dot", [x, y]);
}
function conv$1(x, y, params = {}) {
	if (x.ndim !== y.ndim) throw new Error(`conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`);
	const vmapDims = params.vmapDims ?? 0;
	const n = x.ndim - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	return bind1("conv", [x, y], {
		vmapDims,
		strides: params.strides ?? require_backend.rep(n, 1),
		padding: params.padding ?? require_backend.rep(n, [0, 0]),
		lhsDilation: params.lhsDilation ?? require_backend.rep(n, 1),
		rhsDilation: params.rhsDilation ?? require_backend.rep(n, 1)
	});
}
function compare(x, y, op) {
	return bind1("compare", [x, y], { op });
}
function greater$1(x, y) {
	return compare(y, x, "less");
}
function less$1(x, y) {
	return compare(x, y, "less");
}
function equal$1(x, y) {
	return compare(x, y, "equal");
}
function notEqual$1(x, y) {
	return compare(x, y, "not_equal");
}
function greaterEqual$1(x, y) {
	return compare(y, x, "less_equal");
}
function lessEqual$1(x, y) {
	return compare(x, y, "less_equal");
}
function where$1(cond, x, y) {
	return bind1("where", [
		cond,
		x,
		y
	]);
}
function concatenate$1(xs, axis) {
	if (xs.length === 0) throw new Error("concatenate requires at least one input");
	const avals = xs.map((x) => ShapedArray.fromAval(getAval(x)));
	axis = require_backend.checkAxis(axis, avals[0].ndim);
	for (const x of avals) if (x.ndim !== avals[0].ndim || !x.shape.every((s, i) => i === axis || s === avals[0].shape[i])) throw new Error(`Concatenate: inputs ${avals[0]} and ${x} must match shapes except on axis ${axis}`);
	return bind1("concatenate", xs, { axis });
}
function split$2(x, axis, sizes) {
	axis = require_backend.checkAxis(axis, ndim$1(x));
	if (sizes.some((s) => s < 0 || !Number.isInteger(s))) throw new Error(`split: sizes must be nonnegative integers, got ${JSON.stringify(sizes)}`);
	const totalSize = sizes.reduce((a, b) => a + b, 0);
	if (totalSize !== getShape(x)[axis]) throw new Error(`split: sizes must sum to the size of the axis ${axis}, got ${totalSize}`);
	return bind("split", [x], {
		axis,
		sizes
	});
}
function randomBits(k0, k1, shape, mode = "xor") {
	if (!require_backend.deepEqual(k0.shape, k1.shape) || k0.dtype !== "uint32" || k1.dtype !== "uint32") throw new Error(`randomBits: key parts must be uint32 with the same shape, got ${ShapedArray.fromAval(k0.aval)} and ${ShapedArray.fromAval(k1.aval)}`);
	return bind1("random_bits", [k0, k1], {
		shape,
		mode
	});
}
function normalizeGatherIndexing(operandNdim, indexCount, axis, outDim) {
	if (indexCount === 0) throw new Error("gather() requires at least one index");
	if (!Array.isArray(axis) || axis.length !== indexCount) throw new Error(`Invalid gather() axis: expected ${indexCount} axes, got ${JSON.stringify(axis)}`);
	axis = axis.map((a) => require_backend.checkAxis(a, operandNdim));
	if (new Set(axis).size !== axis.length) throw new Error(`Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`);
	return [axis, require_backend.checkAxis(outDim, operandNdim - axis.length + 1)];
}
function gather(x, indices, axis, outDim, uniqueIndices = false) {
	[axis, outDim] = normalizeGatherIndexing(ndim$1(x), indices.length, axis, outDim);
	return bind1("gather", [x, ...indices], {
		axis,
		outDim,
		uniqueIndices
	});
}
function transpose$1(x, perm) {
	perm = perm ? perm.map((a) => require_backend.checkAxis(a, ndim$1(x))) : require_backend.range(ndim$1(x)).reverse();
	if (!require_backend.isPermutation(perm, ndim$1(x))) throw new Error(`Invalid transpose permutation for ${ndim$1(x)} axes: ${JSON.stringify(perm)}`);
	return bind1("transpose", [x], { perm });
}
function broadcast(x, shape, axis) {
	axis = require_backend.normalizeAxis(axis, shape.length);
	return bind1("broadcast", [x], {
		shape,
		axis
	});
}
function reshape$1(x, shape) {
	if (typeof shape === "number") shape = [shape];
	const originalShape = getShape(x);
	const autoIdx = shape.indexOf(-1);
	if (autoIdx !== -1) {
		const remaining = require_backend.prod(originalShape) / -require_backend.prod(shape);
		if (!Number.isInteger(remaining) || remaining < 0) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`);
		shape = shape.toSpliced(autoIdx, 1, remaining);
	}
	if (require_backend.prod(originalShape) !== require_backend.prod(shape)) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`);
	return bind1("reshape", [x], { shape });
}
function flip$1(x, axis) {
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	return bind1("flip", [x], { axis });
}
function shrink(x, slice) {
	const shape = getShape(x);
	if (!Array.isArray(slice) || !slice.every(require_backend.isNumberPair)) throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
	if (slice.length !== shape.length) throw new Error(`Invalid shrink(): expected ${shape.length} axes, got ${slice.length}`);
	for (let i = 0; i < shape.length; i++) {
		const [start, end] = slice[i];
		if (start > end || start < 0 || end > shape[i]) throw new Error(`Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape[i]}`);
	}
	return bind1("shrink", [x], { slice });
}
function pad$1(x, width) {
	const nd = ndim$1(x);
	let w;
	if (typeof width === "number") w = [[width, width]];
	else if (require_backend.isNumberPair(width)) w = [width];
	else if (!Array.isArray(width)) {
		const indicesAndPairs = Object.entries(width);
		w = require_backend.rep(nd, [0, 0]);
		for (const [k, v] of indicesAndPairs) w[require_backend.checkAxis(parseInt(k), nd)] = v;
	} else if (!width.every(require_backend.isNumberPair)) throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
	else w = width;
	if (w.length === 1) {
		const [w0, w1] = w[0];
		w = require_backend.rep(nd, () => [w0, w1]);
	} else if (w.length !== nd) throw new Error(`Invalid pad(): expected ${nd} axes, got ${w.length}`);
	return bind1("pad", [x], { width: w });
}
function triangularSolve$1(a, b, { lower = false, unitDiagonal = false } = {}) {
	const as = getShape(a);
	const bs = getShape(b);
	if (as.length < 2 || bs.length < 2) throw new Error(`triangular_solve: must be >=2D, got a=${as}, b=${bs}`);
	const n = as[as.length - 2];
	if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
	if (lower) {
		a = flip$1(a, [-2, -1]);
		b = flip$1(b, [-1]);
	}
	let x = bind1("triangular_solve", [a, b], { unitDiagonal });
	if (lower) x = flip$1(x, [-1]);
	return x;
}
function cholesky$2(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`cholesky: expected batch of square matrices, got ${aval}`);
	return bind1("cholesky", [x]);
}
function lu$1(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2) throw new Error(`lu: expected batch of matrices, got ${aval}`);
	return bind("lu", [x]);
}
function jacobiEigh(x, { maxSweeps, tolerance }) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`jacobi_eigh: expected batch of square matrices, got ${aval}`);
	if (!require_backend.isFloatDtype(aval.dtype)) throw new TypeError(`jacobi_eigh: expected floating-point input, got ${aval}`);
	return bind("jacobi_eigh", [x], {
		maxSweeps,
		tolerance
	});
}
function fft$1(real, imag, params) {
	return bind("fft", [real, imag], params);
}
function sort$1(x) {
	if (ndim$1(x) === 0) throw new Error("sort: requires at least 1D input");
	return bind1("sort", [x]);
}
function argsort$1(x) {
	if (ndim$1(x) === 0) throw new Error("argsort: requires at least 1D input");
	return bind("argsort", [x]);
}
function scatter(updates, indices, params) {
	return bind1("scatter", [updates, ...indices], params);
}
function bind1(prim, args, params = {}) {
	const [results] = bind(prim, args, params);
	return results;
}
const traceStack = [];
let dynamicTrace = null;
/**
* Push an interpreter onto the trace stack. Use this like:
* `using main = newMain(...);`
*/
function newMain(traceType, globalData = null) {
	const main = {
		level: traceStack.length,
		traceType,
		globalData
	};
	traceStack.push(main);
	return Object.assign(main, { [Symbol.dispose]() {
		traceStack.pop();
	} });
}
/**
* Set the current dynamic trace, which stashes the current interpreter stack
* and acts temporarily as the bottom of the stack. Use this like:
* `using _dynamic = newDynamic(main);`
*/
function newDynamic(main) {
	const prevDynamicTrace = dynamicTrace;
	dynamicTrace = main;
	return { [Symbol.dispose]() {
		dynamicTrace = prevDynamicTrace;
	} };
}
function currentTraceLevel() {
	return traceStack[traceStack.length - 1].level;
}
var Trace = class {
	main;
	constructor(main) {
		this.main = main;
	}
};
/**
* Broadcast shapes and promote types with casting for two avals.
*
* This implements the weak type behavior described in `promoteTypes()`, but not
* implemented in that function as `weakType` is not passed.
*/
function promoteAvals(a, b) {
	const shape = require_backend.generalBroadcast(a.shape, b.shape);
	const weakType = a.weakType && b.weakType;
	let dtype;
	if (a.weakType === b.weakType) dtype = require_backend.promoteTypes(a.dtype, b.dtype);
	else if (a.weakType) dtype = require_backend.promoteTypes(b.dtype, "uint32");
	else dtype = require_backend.promoteTypes(a.dtype, "uint32");
	return new ShapedArray(shape, dtype, weakType);
}
var Tracer = class Tracer {
	/** @ignore */
	_trace;
	constructor(trace) {
		this._trace = trace;
	}
	/** The shape of the array. */
	get shape() {
		return this.aval.shape;
	}
	/** The total number of elements in the array. */
	get size() {
		return require_backend.prod(this.shape);
	}
	/** The dtype of elements stored in the array. */
	get dtype() {
		return this.aval.dtype;
	}
	/**
	* Whether the array is weakly typed.
	*
	* Weakly typed arrays will cast to the dtype of the other operand. See
	* `promoteTypes()` for details.
	*/
	get weakType() {
		return this.aval.weakType;
	}
	/** The number of dimensions of the array. */
	get ndim() {
		return this.shape.length;
	}
	/** @ignore */
	fullLower() {
		return this;
	}
	neg() {
		return neg(this);
	}
	add(other) {
		return add$1(this, other);
	}
	mul(other) {
		return mul(this, other);
	}
	mod(other) {
		return mod(this, other);
	}
	greater(other) {
		return greater$1(this, other);
	}
	less(other) {
		return less$1(this, other);
	}
	equal(other) {
		return equal$1(this, other);
	}
	notEqual(other) {
		return notEqual$1(this, other);
	}
	greaterEqual(other) {
		return greaterEqual$1(this, other);
	}
	lessEqual(other) {
		return lessEqual$1(this, other);
	}
	/** Sum of the elements of the array over a given axis, or axes. */
	sum(axis = null, opts) {
		return reduce(this, "Add", axis, opts);
	}
	/** Product of the array elements over a given axis. */
	prod(axis = null, opts) {
		return reduce(this, "Mul", axis, opts);
	}
	/** Compute the average of the array elements along the specified axis. */
	mean(axis = null, opts) {
		axis = require_backend.normalizeAxis(axis, this.ndim);
		const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
		if (n === 0) throw new Error("mean: cannot compute mean over zero-length axis");
		const originalDtype = this.dtype;
		const castDtype = require_backend.promoteTypes(originalDtype, "float32");
		const out = reduce(this.astype(castDtype), "Add", axis, opts).mul(1 / n);
		return require_backend.isFloatDtype(originalDtype) ? out.astype(originalDtype) : out;
	}
	/** Minimum of the elements of the array along a given axis. */
	min(axis = null, opts) {
		return reduce(this, "Min", axis, opts);
	}
	/** Maximum of the elements of the array along a given axis. */
	max(axis = null, opts) {
		return reduce(this, "Max", axis, opts);
	}
	/** Test whether all array elements along a given axis evaluate to true. */
	all(axis = null, opts) {
		return this.astype("bool").min(axis, opts);
	}
	/** Test whether any array element along a given axis evaluates to true. */
	any(axis = null, opts) {
		return this.astype("bool").max(axis, opts);
	}
	/** Permute the dimensions of an array. Defaults to reversing the axis order. */
	transpose(perm) {
		return transpose$1(this, perm);
	}
	/**
	* Give a new shape to an array without changing its data.
	*
	* One shape dimension can be -1. In this case, the value is inferred from the
	* length of the array and remaining dimensions.
	*/
	reshape(shape) {
		return reshape$1(this, shape);
	}
	/** Copy the array and cast to a specified dtype. */
	astype(dtype) {
		if (this.dtype === dtype) return this;
		return cast(this, dtype);
	}
	/** Return a bitwise cast of the array, viewed as a new dtype. */
	view(dtype) {
		if (!dtype || dtype === this.dtype) return this;
		return bitcast(this, dtype);
	}
	/** Subtract an array from this one. */
	sub(other) {
		return this.add(neg(other));
	}
	/** Divide an array by this one. */
	div(other) {
		if (require_backend.isFloatDtype(this.dtype)) return this.mul(reciprocal$1(other));
		return idiv(this, other);
	}
	/** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
	diagonal(offset = 0, axis1 = 0, axis2 = 1) {
		if (!Number.isInteger(offset)) throw new TypeError(`offset must be an integer, got ${offset}`);
		if (offset < 0) return this.diagonal(-offset, axis2, axis1);
		axis1 = require_backend.checkAxis(axis1, this.ndim);
		axis2 = require_backend.checkAxis(axis2, this.ndim);
		if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
		if (offset >= this.shape[axis2]) throw new Error("offset exceeds axis size");
		let ar = this;
		if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
			const perm = require_backend.range(ar.ndim).filter((i) => i !== axis1 && i !== axis2).concat(axis1, axis2);
			ar = ar.transpose(perm);
		}
		const [n, m] = ar.shape.slice(-2);
		const diagSize = Math.min(n, m - offset);
		ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
		const npad = diagSize * (m + 1) - n * m;
		if (npad > 0) ar = pad$1(ar, [...require_backend.rep(ar.ndim - 1, [0, 0]), [0, npad]]);
		else if (npad < 0) ar = shrink(ar, [...ar.shape.slice(0, -1), n * m + npad].map((x) => [0, x]));
		ar = ar.reshape([
			...ar.shape.slice(0, -1),
			diagSize,
			m + 1
		]);
		ar = shrink(ar, [...ar.shape.slice(0, -1).map((x) => [0, x]), [offset, offset + 1]]).reshape(ar.shape.slice(0, -1));
		return ar;
	}
	/** Flatten the array without changing its data. */
	flatten() {
		return this.reshape(-1);
	}
	/** Flatten the array without changing its data. */
	ravel() {
		return this.reshape(-1);
	}
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
	*[Symbol.iterator]() {
		if (this.ndim === 0) throw new Error("Cannot iterate over a scalar array");
		let residual = this;
		const subarrayShape = this.shape.slice(1);
		for (let i = 0; i < this.shape[0]; i++) {
			const lr = split$2(residual, 0, [1, residual.shape[0] - 1]);
			yield lr[0].reshape(subarrayShape);
			residual = lr[1];
		}
		residual.dispose();
	}
	/**
	* Return a sorted copy of an array in ascending order.
	*
	* See `jax.numpy.sort` for full docs.
	*/
	sort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (this.shape[axis] <= 1) return this;
		if (axis === this.ndim - 1) return sort$1(this);
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		return sort$1(this.transpose(perm)).transpose(require_backend.invertPermutation(perm));
	}
	/**
	* Return the indices that would sort an array. Unlike `sort`, this is
	* guaranteed to be a stable sorting algorithm; it always returns the smaller
	* index first in event of ties.
	*
	* See `jax.numpy.argsort` for full docs.
	*/
	argsort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (axis === this.ndim - 1) {
			const [y, yi] = argsort$1(this);
			y.dispose();
			return yi;
		}
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		const [y, yi] = argsort$1(this.transpose(perm));
		y.dispose();
		return yi.transpose(require_backend.invertPermutation(perm));
	}
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
	slice(...index) {
		const checkBounds = (n, i) => {
			if (i > n || i < -n) throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
			return i < 0 ? n + i : i;
		};
		const hasAdvancedIdx = index.some((value) => value instanceof Tracer);
		const axesForGather = [];
		let outDim = -1;
		if (hasAdvancedIdx) {
			const advancedAxes = [];
			let currentAxisForGather = 0;
			for (let i = 0; i < index.length; i++) {
				const value = index[i];
				if (value instanceof Tracer) {
					advancedAxes.push(i);
					axesForGather.push(currentAxisForGather++);
				} else if (typeof value === "number") advancedAxes.push(i);
				else currentAxisForGather++;
			}
			if (advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !== advancedAxes.length - 1) outDim = 0;
			else outDim = axesForGather[0];
		}
		const slice = [];
		const basicShape = [];
		let needsReshape = false;
		let axis = 0;
		for (const value of index) if (value === null) {
			basicShape.push(1);
			needsReshape = true;
		} else if (typeof value === "number") {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const i = checkBounds(this.shape[axis++], value);
			slice.push([i, i + 1]);
			needsReshape = true;
		} else if (Array.isArray(value)) {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const n = this.shape[axis++];
			if (value.length === 0) {
				basicShape.push(n);
				slice.push([0, n]);
			} else if (value.length === 1) {
				const i = checkBounds(n, value[0]);
				basicShape.push(n - i);
				slice.push([i, n]);
			} else if (value.length === 2) {
				const [i, j] = value.map((v) => checkBounds(n, v));
				if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
				basicShape.push(j - i);
				slice.push([i, j]);
			}
		} else if (value instanceof Tracer) {
			const n = this.shape[axis++];
			basicShape.push(n);
			slice.push([0, n]);
		} else throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
		while (axis < this.shape.length) {
			slice.push([0, this.shape[axis]]);
			basicShape.push(this.shape[axis++]);
		}
		let result = shrink(this, slice);
		result = needsReshape ? reshape$1(result, basicShape) : result;
		if (hasAdvancedIdx) result = gather(result, index.filter((a) => a instanceof Tracer), axesForGather, outDim);
		return result;
	}
};
function ndim$1(x) {
	if (x instanceof Tracer) return x.shape.length;
	else return 0;
}
function getShape(x) {
	return x instanceof Tracer ? x.shape : [];
}
var ShapedArray = class ShapedArray {
	shape;
	dtype;
	weakType;
	constructor(shape, dtype, weakType) {
		this.shape = shape;
		this.dtype = dtype;
		this.weakType = weakType;
	}
	static fromAval(aval) {
		return new ShapedArray(aval.shape, aval.dtype, aval.weakType);
	}
	get ndim() {
		return this.shape.length;
	}
	get size() {
		return require_backend.prod(this.shape);
	}
	scalar() {
		return new ShapedArray([], this.dtype, this.weakType);
	}
	toString() {
		return `${this.dtype}[${this.shape.join(",")}]`;
	}
	equals(other) {
		return this === other || this.constructor === other.constructor && this.ndim === other.ndim && this.shape.every((d, i) => d === other.shape[i]);
	}
};
function getAval(x) {
	if (x instanceof Tracer) return x.aval;
	else if (typeof x === "boolean" || typeof x === "number") return new ShapedArray([], typeof x === "boolean" ? "bool" : "float32", typeof x === "boolean" ? false : true);
	else throw new TypeError(`Unknown value: ${x}`);
}
function bind(prim, args, params = {}) {
	const topTrace = findTopTrace(args);
	const tracers = args.map((arg) => fullRaise(topTrace, arg));
	const outs = topTrace.processPrimitive(prim, tracers, params);
	if (require_backend.DEBUG >= 5) console.info(`processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`);
	return outs.map((out) => out.fullLower());
}
function findTopTrace(xs) {
	let topMain = traceStack[0];
	for (const x of xs) if (x instanceof Tracer && x._trace.main.level > topMain.level) topMain = x._trace.main;
	if (dynamicTrace && dynamicTrace.level > topMain.level) topMain = dynamicTrace;
	return new topMain.traceType(topMain);
}
function fullRaise(trace, val) {
	if (!(val instanceof Tracer)) return trace.pure(val);
	const level = trace.main.level;
	if (Object.is(val._trace.main, trace.main)) return val;
	else if (val._trace.main.level < level) return trace.lift(val);
	else if (val._trace.main.level > level) throw new Error(`Can't lift Tracer level ${val._trace.main.level} to level ${level}`);
	else throw new Error(`Different traces at same level: ${val._trace.constructor}, ${trace.constructor}.`);
}
var TreeMismatchError = class extends TypeError {
	constructor(where, left, right) {
		super(`Mismatched tree structures in ${where}: ${left} != ${right}`);
	}
};
/** Flatten a function of `JsTree` input/output for use in tracing. */
function flattenFun(f, inTree) {
	const store = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const [outFlat, outTree] = flatten(f(...unflatten(inTree, argsFlat)));
		store.value = outTree;
		return outFlat;
	};
	return [flatFun, store];
}
/** Like flattenFun, but expects f to return [main, aux] tuple. */
function flattenFunWithAux(f, inTree) {
	const store = { value: void 0 };
	const auxStore = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const result = f(...unflatten(inTree, argsFlat));
		if (!Array.isArray(result) || result.length !== 2) throw new Error("Function with `hasAux: true` must return [output, aux] tuple");
		const [out, aux] = result;
		const [outFlat, outTree] = flatten(out);
		store.value = outTree;
		auxStore.value = aux;
		return outFlat;
	};
	return [
		flatFun,
		store,
		auxStore
	];
}
var UseAfterFreeError = class extends ReferenceError {
	constructor(tracer) {
		super(`Referenced tracer ${tracer.toString()} freed, please use .ref move semantics`);
	}
};
//#endregion
//#region \0@oxc-project+runtime@0.137.0/helpers/esm/usingCtx.js
function _usingCtx() {
	var r = "function" == typeof SuppressedError ? SuppressedError : function(r, e) {
		var n = Error();
		return n.name = "SuppressedError", n.error = r, n.suppressed = e, n;
	}, e = {}, n = [];
	function using(r, e) {
		if (null != e) {
			if (Object(e) !== e) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
			if (r) var o = e[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
			if (void 0 === o && (o = e[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r)) var t = o;
			if ("function" != typeof o) throw new TypeError("Object is not disposable.");
			t && (o = function o() {
				try {
					t.call(e);
				} catch (r) {
					return Promise.reject(r);
				}
			}), n.push({
				v: e,
				d: o,
				a: r
			});
		} else r && n.push({
			d: e,
			a: r
		});
		return e;
	}
	return {
		e,
		u: using.bind(null, !1),
		a: using.bind(null, !0),
		d: function d() {
			var o, t = this.e, s = 0;
			function next() {
				for (; o = n.pop();) try {
					if (!o.a && 1 === s) return s = 0, n.push(o), Promise.resolve().then(next);
					if (o.d) {
						var r = o.d.call(o.v);
						if (o.a) return s |= 2, Promise.resolve(r).then(next, err);
					} else s |= 1;
				} catch (r) {
					return err(r);
				}
				if (1 === s) return t !== e ? Promise.reject(t) : Promise.resolve();
				if (t !== e) throw t;
			}
			function err(n) {
				return t = t !== e ? new r(n, t) : n, next();
			}
			return next();
		}
	};
}
//#endregion
//#region src/frontend/jaxpr.ts
/** Variable in a Jaxpr expression. */
var Var = class Var {
	static #nextId = 1;
	id;
	aval;
	constructor(aval) {
		this.id = Var.#nextId++;
		this.aval = aval;
	}
	toString() {
		return `Var(${this.id}):${this.aval.toString()}`;
	}
};
/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
var Lit = class {
	value;
	aval;
	get dtype() {
		return this.aval.dtype;
	}
	constructor(aval, value) {
		if (aval.shape.length !== 0) throw new Error(`internal: Lit must be a scalar`);
		this.value = value;
		this.aval = ShapedArray.fromAval(aval);
	}
};
function atomIsLit(atom, literal) {
	return atom instanceof Lit && (literal === void 0 || atom.value === literal);
}
var VarPrinter = class {
	names = /* @__PURE__ */ new Map();
	#next = "a";
	#advance() {
		const ret = this.#next;
		let lastNonz = this.#next.length - 1;
		while (lastNonz >= 0 && this.#next[lastNonz] === "z") lastNonz--;
		if (lastNonz < 0) this.#next = "a".repeat(this.#next.length + 1);
		else {
			let result = this.#next.slice(0, lastNonz);
			result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
			result += "a".repeat(this.#next.length - 1 - lastNonz);
			this.#next = result;
		}
		return ret;
	}
	name(v) {
		if (this.names.has(v)) return this.names.get(v);
		const name = this.#advance();
		this.names.set(v, name);
		return name;
	}
	nameType(v) {
		return `${this.name(v)}:${v.aval.toString()}`;
	}
};
/** A single statement / binding in a Jaxpr, in ANF form. */
var JaxprEqn = class {
	primitive;
	inputs;
	params;
	outBinders;
	constructor(primitive, inputs, params, outBinders) {
		this.primitive = primitive;
		this.inputs = inputs;
		this.params = params;
		this.outBinders = outBinders;
	}
	pprint(usedVars, vp = new VarPrinter()) {
		const lhs = require_backend.PPrint.pp(this.outBinders.map((v) => !usedVars || usedVars.has(v) ? vp.nameType(v) : "_").join(" "));
		let rhs = require_backend.PPrint.pp(this.primitive);
		const paramsList = Object.entries(this.params).map(([k, v]) => require_backend.PPrint.pp(`${k}=${v}`));
		if (paramsList.length > 0) rhs = rhs.stack(require_backend.PPrint.pp(" [ ")).stack(require_backend.PPrint.prototype.concat(...paramsList)).stack(require_backend.PPrint.pp(" ] "));
		else rhs = rhs.stack(require_backend.PPrint.pp(" "));
		rhs = rhs.stack(require_backend.PPrint.pp(this.inputs.map((x) => x instanceof Var ? vp.name(x) : String(x.value)).join(" ")));
		return lhs.stack(require_backend.PPrint.pp(" = ")).stack(rhs);
	}
	toString() {
		return this.pprint().toString();
	}
};
/** Typed intermediate representation for traced computations. */
var Jaxpr = class Jaxpr {
	inBinders;
	eqns;
	outs;
	#hash;
	constructor(inBinders, eqns, outs) {
		this.inBinders = inBinders;
		this.eqns = eqns;
		this.outs = outs;
	}
	pprint() {
		const vp = new VarPrinter();
		const usedVars = new Set([...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter((x) => x instanceof Var));
		const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
		const eqns = require_backend.PPrint.prototype.concat(...this.eqns.map((e) => e.pprint(usedVars, vp)));
		const outs = this.outs.map((x) => x instanceof Var ? vp.name(x) : x.value).join(", ");
		return require_backend.PPrint.pp(`{ lambda ${inBinders} .`).concat((this.eqns.length ? require_backend.PPrint.pp("let ").stack(eqns).concat(require_backend.PPrint.pp(`in ( ${outs} ) }`)) : require_backend.PPrint.pp(`( ${outs} ) }`)).indent(2));
	}
	toString() {
		return this.pprint().toString();
	}
	/**
	* Gets a hash of this Jaxpr.
	*
	* Var identity is not considered in the hash, so two Jaxprs with the same
	* order of assignments and operators but different variable IDs will resolve
	* to the same hash (and toString representation).
	*/
	getHash() {
		if (this.#hash !== void 0) return this.#hash;
		const hasher = new require_backend.FpHash();
		const varIds = /* @__PURE__ */ new Map();
		const vi = (v) => {
			if (varIds.has(v)) return varIds.get(v);
			const id = varIds.size + 1;
			varIds.set(v, require_backend.FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
			return id;
		};
		hasher.update(this.inBinders.length);
		for (const x of this.inBinders) hasher.update(vi(x));
		hasher.update(this.eqns.length);
		for (const eqn of this.eqns) {
			hasher.update(eqn.primitive);
			hasher.update(eqn.inputs.length);
			for (const x of eqn.inputs) hasher.update(x instanceof Var ? vi(x) : x.value);
			hasher.update(JSON.stringify(eqn.params));
			hasher.update(eqn.outBinders.length);
			for (const x of eqn.outBinders) hasher.update(vi(x));
		}
		hasher.update(this.outs.length);
		for (const x of this.outs) hasher.update(x instanceof Var ? vi(x) : x.value);
		return this.#hash = hasher.value;
	}
	hash(state) {
		state.update(this.getHash());
	}
	/**
	* Produce a simplified Jaxpr with basic optimizations applied.
	*  - Trim away unused variables.
	*  - Fold away *1, *0, or +0 operations against literals.
	*  - Remove no-op movement operations.
	*/
	simplify() {
		const context = /* @__PURE__ */ new Map();
		const newEqns = [];
		for (const e of this.eqns) {
			const inputs = e.inputs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
			const eqn = new JaxprEqn(e.primitive, inputs, e.params, e.outBinders);
			if (eqn.primitive === "add") {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 0)) context.set(c, b);
				else if (atomIsLit(b, 0)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.dtype === "bool" ? Math.min(a.value + b.value, 1) : a.value + b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === "neg") {
				const [a] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a)) context.set(c, new Lit(a.aval, -a.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === "mul") {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 1)) context.set(c, b);
				else if (atomIsLit(b, 1)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.value * b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === "idiv") {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(b, 1) && !require_backend.isFloatDtype(a.aval.dtype)) context.set(c, a);
				else newEqns.push(eqn);
			} else if ((eqn.primitive === "broadcast" || eqn.primitive === "reshape") && require_backend.deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape) || eqn.primitive === "transpose" && eqn.params.perm.every((p, i) => p === i) || eqn.primitive === "flip" && eqn.params.axis.length === 0 || eqn.primitive === "shrink" && eqn.params.slice.every(([s, e], i) => s === 0 && e === eqn.inputs[0].aval.shape[i]) || eqn.primitive === "pad" && eqn.params.width.every(([w0, w1]) => w0 === 0 && w1 === 0)) context.set(eqn.outBinders[0], eqn.inputs[0]);
			else newEqns.push(eqn);
		}
		const outs = this.outs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
		const usedVars = new Set(outs.filter((x) => x instanceof Var));
		const liveEqns = [];
		for (let i = newEqns.length - 1; i >= 0; i--) {
			const eqn = newEqns[i];
			if (eqn.outBinders.some((v) => usedVars.has(v))) {
				liveEqns.push(eqn);
				for (const v of eqn.inputs) if (v instanceof Var) usedVars.add(v);
			}
		}
		return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
	}
	/** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
	flatten() {
		if (!this.eqns.some((eqn) => eqn.primitive === "jit")) return this;
		const newEqns = [];
		const varMap = /* @__PURE__ */ new Map();
		const varMapF = (x) => x instanceof Var ? varMap.get(x) ?? x : x;
		for (const eqn of this.eqns) if (eqn.primitive === "jit") {
			const jaxpr = eqn.params.jaxpr.flatten();
			const translation = /* @__PURE__ */ new Map();
			const translationF = (x) => x instanceof Var ? translation.get(x) : x;
			for (const [v, x] of require_backend.zip(jaxpr.inBinders, eqn.inputs)) translation.set(v, varMapF(x));
			for (const ieqn of jaxpr.eqns) {
				const inputs = ieqn.inputs.map(translationF);
				const outBinders = [];
				for (const v of ieqn.outBinders) {
					const u = new Var(v.aval);
					outBinders.push(u);
					translation.set(v, u);
				}
				newEqns.push(new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders));
			}
			for (const [v, x] of require_backend.zip(eqn.outBinders, jaxpr.outs)) varMap.set(v, translationF(x));
		} else if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) newEqns.push(new JaxprEqn(eqn.primitive, eqn.inputs.map(varMapF), eqn.params, eqn.outBinders));
		else newEqns.push(eqn);
		const newOuts = this.outs.map(varMapF);
		return new Jaxpr(this.inBinders, newEqns, newOuts);
	}
};
var JaxprType = class {
	inTypes;
	outTypes;
	constructor(inTypes, outTypes) {
		this.inTypes = inTypes;
		this.outTypes = outTypes;
	}
	toString() {
		return `(${this.inTypes.map((aval) => aval.toString()).join(", ")}) -> (${this.outTypes.map((aval) => aval.toString()).join(", ")})`;
	}
};
function typecheckJaxpr(jaxpr) {
	const env = /* @__PURE__ */ new Set();
	for (const v of jaxpr.inBinders) {
		if (env.has(v)) throw new TypeError(`Duplicate variable binding: ${v}`);
		env.add(v);
	}
	for (const eqn of jaxpr.eqns) {
		const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
		const rule = abstractEvalRules[eqn.primitive];
		const outTypes = rule(inTypes, eqn.params);
		for (const [outBinder, outType] of require_backend.zip(eqn.outBinders, outTypes)) {
			if (!outType.equals(outBinder.aval)) throw new TypeError(`Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`);
			if (env.has(outBinder)) throw new TypeError(`Duplicate variable binding: ${outBinder}`);
			env.add(outBinder);
		}
	}
	return new JaxprType(jaxpr.inBinders.map((v) => v.aval), jaxpr.outs.map((x) => typecheckAtom(env, x)));
}
function typecheckAtom(env, x) {
	if (x instanceof Var) {
		if (!env.has(x)) throw new Error(`Unknown variable: ${x}`);
		return x.aval;
	} else if (x instanceof Lit) return x.aval;
	else throw new TypeError(`Invalid atom type: ${x}`);
}
/** Evaluate a Jaxpr on an array of inputs. */
function evalJaxpr(jaxpr, args) {
	const env = /* @__PURE__ */ new Map();
	const usageCount = /* @__PURE__ */ new Map();
	for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
	const remainingRefs = /* @__PURE__ */ new Map();
	const read = (x) => {
		if (x instanceof Var) {
			remainingRefs.set(x, (remainingRefs.get(x) ?? 0) - 1);
			return env.get(x);
		} else return array(x.value, { dtype: x.dtype });
	};
	const write = (v, val) => {
		if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
		let refCount = usageCount.get(v) ?? 0;
		if (refCount) {
			env.set(v, val);
			remainingRefs.set(v, refCount);
			while (refCount-- > 1) val.ref;
		} else val.dispose();
	};
	try {
		for (const [v, arg] of require_backend.zip(jaxpr.inBinders, args)) write(v, arg);
		for (const eqn of jaxpr.eqns) {
			const inVals = eqn.inputs.map(read);
			const outVals = bind(eqn.primitive, inVals, eqn.params);
			for (const [v, val] of require_backend.zip(eqn.outBinders, outVals)) write(v, val);
		}
		return jaxpr.outs.map(read);
	} catch (error) {
		for (let [v, refCount] of remainingRefs.entries()) if (refCount > 0) {
			const tracer = env.get(v);
			while (refCount--) tracer.dispose();
		}
		throw error;
	}
}
/** Convert a Jaxpr to a callable function by evaluating it. */
function jaxprAsFun(jaxpr) {
	return (...args) => evalJaxpr(jaxpr, args);
}
/** Jaxpr with a collection of associated, traced constants. */
var ClosedJaxpr = class ClosedJaxpr {
	jaxpr;
	consts;
	constructor(jaxpr, consts) {
		this.jaxpr = jaxpr;
		this.consts = consts;
	}
	/** String representation of this Jaxpr. */
	toString() {
		return this.jaxpr.toString();
	}
	/** Apply a function to the underlying Jaxpr. */
	mapJaxpr(f) {
		return new ClosedJaxpr(f(this.jaxpr), this.consts);
	}
	/** Dispose of the constants in this Jaxpr. */
	dispose() {
		for (const c of this.consts) c.dispose();
	}
};
/** Tracer that records its operations to dynamically construct a Jaxpr. */
var JaxprTracer = class extends Tracer {
	aval;
	#rc;
	constructor(trace, aval) {
		super(trace);
		this.aval = aval;
		this.#rc = 1;
	}
	toString() {
		return `JaxprTracer(${this.aval.toString()})`;
	}
	get ref() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc++;
		return this;
	}
	dispose() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc--;
	}
	trackLiftedConstant() {
		this.#rc++;
	}
};
/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
var JaxprTrace = class extends Trace {
	/** Register a Jaxpr argument with a given shape and return the tracer. */
	newArg(aval) {
		aval = ShapedArray.fromAval(aval);
		const tracer = this.builder.newTracer(this, aval);
		this.builder.addVar(tracer);
		return tracer;
	}
	/** Register a constant / literal in this Jaxpr. */
	getOrMakeConstTracer(val) {
		if (!(val instanceof Tracer)) val = pureArray(val);
		let tracer = this.builder.constTracers.get(val);
		if (tracer === void 0) {
			tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
			this.builder.addConst(tracer, val);
		} else {
			val.dispose();
			tracer.trackLiftedConstant();
		}
		return tracer;
	}
	pure = this.getOrMakeConstTracer;
	lift = this.getOrMakeConstTracer;
	processPrimitive(primitive, tracers, params) {
		const avalsIn = tracers.map((t) => {
			t.dispose();
			return t.aval;
		});
		const outTracers = abstractEvalRules[primitive](avalsIn, params).map((aval) => this.builder.newTracer(this, aval));
		this.builder.addEqn(new JaxprEqn(primitive, tracers.map((t) => this.builder.getVar(t)), params, outTracers.map((t) => this.builder.addVar(t))));
		return outTracers;
	}
	get builder() {
		return this.main.globalData;
	}
};
/** Incrementally constructs a Jaxpr. */
var JaxprBuilder = class {
	eqns = [];
	tracerToVar = /* @__PURE__ */ new Map();
	constTracers = /* @__PURE__ */ new Map();
	constVals = /* @__PURE__ */ new Map();
	tracers = [];
	newTracer(trace, aval) {
		const tracer = new JaxprTracer(trace, aval);
		this.tracers.push(tracer);
		return tracer;
	}
	addEqn(eqn) {
		this.eqns.push(eqn);
	}
	addVar(tracer) {
		if (this.tracerToVar.has(tracer)) throw new Error(`Tracer was added as variable twice: ${tracer}`);
		const v = new Var(tracer.aval);
		this.tracerToVar.set(tracer, v);
		return v;
	}
	getVar(tracer) {
		const v = this.tracerToVar.get(tracer);
		if (v === void 0) throw new Error(`Could not find variable for tracer: ${tracer}`);
		return v;
	}
	addConst(tracer, val) {
		const v = this.addVar(tracer);
		this.constTracers.set(val, tracer);
		this.constVals.set(v, val);
		return v;
	}
	build(inTracers, outTracers) {
		const [constVars, consts] = require_backend.unzip2(this.constVals.entries());
		const t2v = this.getVar.bind(this);
		const inBinders = [...constVars, ...inTracers.map(t2v)];
		const outVars = outTracers.map(t2v);
		const jaxpr = new Jaxpr(inBinders, this.eqns, outVars);
		typecheckJaxpr(jaxpr);
		return _inlineLiterals(new ClosedJaxpr(jaxpr, consts));
	}
};
function _inlineLiterals({ jaxpr, consts }) {
	const literals = /* @__PURE__ */ new Map();
	const constBinders = [];
	const newConsts = [];
	for (let i = 0; i < consts.length; i++) if (ndim$1(consts[i]) === 0 && consts[i] instanceof Array$1) {
		const ar = consts[i];
		literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
	} else {
		constBinders.push(jaxpr.inBinders[i]);
		newConsts.push(consts[i]);
	}
	const newEqns = jaxpr.eqns.map((eqn) => new JaxprEqn(eqn.primitive, eqn.inputs.map((x) => literals.get(x) ?? x), eqn.params, eqn.outBinders));
	const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
	const newJaxpr = new Jaxpr([...constBinders, ...jaxpr.inBinders.slice(consts.length)], newEqns, newOuts);
	typecheckJaxpr(newJaxpr);
	return new ClosedJaxpr(newJaxpr, newConsts);
}
function binopAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("binopAbstractEval expects ShapedArray inputs");
	return [promoteAvals(x, y)];
}
function compareAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("compareAbstractEval expects ShapedArray inputs");
	return [new ShapedArray(promoteAvals(x, y).shape, "bool", false)];
}
function vectorizedUnopAbstractEval([x]) {
	return [ShapedArray.fromAval(x)];
}
/** Shape for the output of a gather operation. */
function gatherShape(operandShape, indices, { axis, outDim }) {
	for (const a of indices) if (a.dtype !== "int32" && a.dtype !== "uint32") throw new TypeError(`indices must be Int32 or Uint32, got ${a.dtype}`);
	if (axis.length !== indices.length) throw new TypeError(`got ${axis} axes but ${indices.length} indices`);
	if (indices.length === 0) throw new TypeError(`must have at least one index`);
	if (axis.some((a) => a < 0 || a >= operandShape.length)) throw new TypeError(`axis out of bounds`);
	if (outDim < 0 || outDim > operandShape.length - axis.length) throw new TypeError(`outDim out of bounds`);
	const axisSet = new Set(axis);
	if (axisSet.size !== axis.length) throw new TypeError(`axes are not unique`);
	const indexShape = indices.reduce((result, a) => require_backend.generalBroadcast(result, a.shape), []);
	const result = operandShape.filter((_, i) => !axisSet.has(i));
	result.splice(outDim, 0, ...indexShape);
	return result;
}
const abstractEvalRules = {
	["add"]: binopAbstractEval,
	["mul"]: binopAbstractEval,
	["idiv"]: binopAbstractEval,
	["mod"]: binopAbstractEval,
	["min"]: binopAbstractEval,
	["max"]: binopAbstractEval,
	["bit_combine"]([x, y]) {
		const aval = promoteAvals(x, y);
		if (require_backend.isFloatDtype(aval.dtype)) throw new TypeError(`bitwise operations require integer or boolean inputs, got ${aval.dtype}`);
		return [aval];
	},
	["bit_shift"]([x, y]) {
		const shape = require_backend.generalBroadcast(x.shape, y.shape);
		if (require_backend.isFloatDtype(x.dtype) || require_backend.isFloatDtype(y.dtype) || x.dtype === "bool" || y.dtype === "bool") throw new TypeError(`bit shift operations require integer inputs, got ${x} and ${y}`);
		return [new ShapedArray(shape, x.dtype, x.weakType)];
	},
	["bit_count"]([x]) {
		if (x.dtype !== "int32" && x.dtype !== "uint32") throw new TypeError(`bit count requires an int32 or uint32 input, got ${x.dtype}`);
		return [new ShapedArray(x.shape, "int32", false)];
	},
	["neg"]: vectorizedUnopAbstractEval,
	["reciprocal"]: vectorizedUnopAbstractEval,
	["floor"]: vectorizedUnopAbstractEval,
	["ceil"]: vectorizedUnopAbstractEval,
	["stop_gradient"]: vectorizedUnopAbstractEval,
	["cast"]([x], { dtype }) {
		return [new ShapedArray(x.shape, dtype, false)];
	},
	["bitcast"]([x], { dtype }) {
		if (x.dtype !== dtype && (x.dtype === "bool" || dtype === "bool")) throw new TypeError("Bitcast to/from bool is not allowed");
		if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
		return [new ShapedArray(x.shape, dtype, false)];
	},
	["sin"]: vectorizedUnopAbstractEval,
	["cos"]: vectorizedUnopAbstractEval,
	["asin"]: vectorizedUnopAbstractEval,
	["atan"]: vectorizedUnopAbstractEval,
	["exp"]: vectorizedUnopAbstractEval,
	["log"]: vectorizedUnopAbstractEval,
	["erf"]: vectorizedUnopAbstractEval,
	["erfc"]: vectorizedUnopAbstractEval,
	["sqrt"]: vectorizedUnopAbstractEval,
	["reduce"]([x], { axis }) {
		const axisSet = new Set(axis);
		return [new ShapedArray(x.shape.filter((_, i) => !axisSet.has(i)), x.dtype, x.weakType)];
	},
	["pool"]([x], { window, strides }) {
		return [new ShapedArray(checkPoolShape(x.shape, window, strides), x.dtype, x.weakType)];
	},
	["pool_transpose"]([x], { inShape, window, strides }) {
		const shape = checkPoolShape(inShape, window, strides);
		if (!require_backend.deepEqual(shape, x.shape)) throw new TypeError(`PoolTranspose shape mismatch: expected ${JSON.stringify(shape)}, got ${JSON.stringify(x.shape)}`);
		return [new ShapedArray(inShape, x.dtype, x.weakType)];
	},
	["dot"]([x, y]) {
		if (x.ndim === 0 && y.ndim === 0) throw new TypeError("Dot requires at least 1D inputs");
		const { shape, dtype, weakType } = promoteAvals(x, y);
		shape.splice(-1, 1);
		return [new ShapedArray(shape, dtype, weakType)];
	},
	["conv"]([lhs, rhs], params) {
		const { dtype, weakType } = promoteAvals(lhs.scalar(), rhs.scalar());
		return [new ShapedArray(checkConvShape(lhs.shape, rhs.shape, params), dtype, weakType)];
	},
	["compare"]: compareAbstractEval,
	["where"]([cond, x, y]) {
		if (cond.dtype !== "bool") throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
		const xy = promoteAvals(x, y);
		return [new ShapedArray(require_backend.generalBroadcast(cond.shape, xy.shape), xy.dtype, xy.weakType)];
	},
	["concatenate"](xs, { axis }) {
		if (xs.length === 0) throw new TypeError("Concatenate requires at least one input");
		for (const x of xs) if (x.ndim !== xs[0].ndim || !x.shape.every((s, i) => i === axis || s === xs[0].shape[i])) throw new TypeError(`Concatenate: inputs ${xs[0]} and ${x} must match shapes except on axis ${axis}`);
		const shape = xs[0].shape.slice();
		shape[axis] = xs.reduce((sum, x) => sum + x.shape[axis], 0);
		const { dtype, weakType } = xs.map((x) => x.scalar()).reduce(promoteAvals);
		return [new ShapedArray(shape, dtype, weakType)];
	},
	["split"]([x], { axis, sizes }) {
		const totalSize = sizes.reduce((a, b) => a + b, 0);
		if (x.shape[axis] !== totalSize) throw new TypeError(`Split: sizes ${sizes} do not sum to dimension ${x.shape[axis]} on axis ${axis}`);
		return sizes.map((size) => {
			return new ShapedArray(x.shape.toSpliced(axis, 1, size), x.dtype, x.weakType);
		});
	},
	["random_bits"]([k0, k1], { shape }) {
		if (k0.dtype !== "uint32" || k1.dtype !== "uint32") throw new TypeError(`RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`);
		if (!require_backend.deepEqual(k0.shape, k1.shape)) throw new TypeError(`RandomBits: Keys have different shapes ${k0.shape} and ${k1.shape}`);
		if (!require_backend.deepEqual(shape.slice(0, k0.ndim), k0.shape)) throw new TypeError(`RandomBits: generated shape ${shape} must match key shape ${k0.shape}`);
		return [new ShapedArray(shape, "uint32", false)];
	},
	["gather"]([x, ...indices], { axis, outDim }) {
		return [new ShapedArray(gatherShape(x.shape, indices, {
			axis,
			outDim
		}), x.dtype, x.weakType)];
	},
	["transpose"]([x], { perm }) {
		return [new ShapedArray(perm.map((i) => x.shape[i]), x.dtype, x.weakType)];
	},
	["broadcast"]([x], { shape }) {
		return [new ShapedArray(shape, x.dtype, x.weakType)];
	},
	["reshape"]([x], { shape }) {
		return [new ShapedArray(shape, x.dtype, x.weakType)];
	},
	["flip"]([x], _) {
		return [ShapedArray.fromAval(x)];
	},
	["shrink"]([x], { slice }) {
		return [new ShapedArray(slice.map((s) => s[1] - s[0]), x.dtype, x.weakType)];
	},
	["pad"]([x], { width }) {
		return [new ShapedArray(x.shape.map((dim, i) => dim + width[i][0] + width[i][1]), x.dtype, x.weakType)];
	},
	["sort"]([x]) {
		if (x.ndim === 0) throw new TypeError("sort: requires at least 1D input");
		return [ShapedArray.fromAval(x)];
	},
	["argsort"]([x]) {
		if (x.ndim === 0) throw new TypeError("argsort: requires at least 1D input");
		return [ShapedArray.fromAval(x), new ShapedArray(x.shape, "int32", false)];
	},
	["scatter"]([updates, ...indices], { shape, axis, outDim }) {
		const expectedUpdatesShape = gatherShape(shape, indices, {
			axis,
			outDim
		});
		if (!require_backend.deepEqual(updates.shape, expectedUpdatesShape)) throw new TypeError(`Scatter updates shape ${updates.shape} does not match expected shape ${expectedUpdatesShape}`);
		return [new ShapedArray(shape, updates.dtype, updates.weakType)];
	},
	["triangular_solve"]([a, b]) {
		if (a.ndim < 2) throw new TypeError(`triangular_solve: a must be at least 2D, got ${a}`);
		if (b.ndim < 2) throw new TypeError(`triangular_solve: b must be at least 2D, got ${b}`);
		const [m, n] = a.shape.slice(-2);
		const [_batch, q] = b.shape.slice(-2);
		if (!require_backend.deepEqual(a.shape.slice(0, -2), b.shape.slice(0, -2)) || a.dtype !== b.dtype || m !== n || n !== q) throw new TypeError(`triangular_solve: mismatch ${a} vs ${b}`);
		return [new ShapedArray(b.shape, b.dtype, a.weakType && b.weakType)];
	},
	["cholesky"]([a]) {
		if (a.ndim < 2) throw new TypeError(`cholesky: requires at least 2D input, got ${a}`);
		if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`cholesky: must be square, got ${a}`);
		return [ShapedArray.fromAval(a)];
	},
	["lu"]([a]) {
		if (a.ndim < 2) throw new TypeError(`lu: requires at least 2D input, got ${a}`);
		const batch = a.shape.slice(0, -2);
		const [m, n] = a.shape.slice(-2);
		return [
			ShapedArray.fromAval(a),
			new ShapedArray([...batch, Math.min(m, n)], "int32", false),
			new ShapedArray([...batch, m], "int32", false)
		];
	},
	["jacobi_eigh"]([a]) {
		if (a.ndim < 2) throw new TypeError(`jacobi_eigh: requires at least 2D input, got ${a}`);
		if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`jacobi_eigh: must be square, got ${a}`);
		if (!require_backend.isFloatDtype(a.dtype)) throw new TypeError(`jacobi_eigh: requires floating-point input, got ${a}`);
		return [ShapedArray.fromAval(a), ShapedArray.fromAval(a)];
	},
	["fft"]([real, imag], { factors }) {
		if (!real.equals(imag)) throw new TypeError(`fft: real and imag arrays must match, got ${real} and ${imag}`);
		if (real.ndim < 1) throw new TypeError(`fft: requires at least 1D input, got ${real}`);
		if (!require_backend.isFloatDtype(real.dtype)) throw new TypeError(`fft: requires floating-point input, got ${real}`);
		const n = real.shape[real.ndim - 1];
		if (require_backend.prod(factors) !== n) throw new TypeError(`fft: factorization ${factors} does not match size ${n}`);
		return [ShapedArray.fromAval(real), ShapedArray.fromAval(real)];
	},
	["jit"](args, { jaxpr }) {
		const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
		if (args.length !== inTypes.length) throw new TypeError(`jit expected ${inTypes.length} arguments, got ${args.length}`);
		for (let i = 0; i < inTypes.length; i++) if (!args[i].equals(inTypes[i])) throw new TypeError(`jit argument ${i} has type ${args[i]}, expected ${inTypes[i]}`);
		return outTypes;
	}
};
function splitIdx(values, argnums) {
	const a = [];
	const b = [];
	for (let i = 0; i < values.length; i++) if (argnums.has(i)) a.push(values[i]);
	else b.push(values[i]);
	return [a, b];
}
function joinIdx(n, a, b, argnums) {
	const result = [];
	let ai = 0;
	let bi = 0;
	for (let i = 0; i < n; i++) if (argnums.has(i)) result.push(a[ai++]);
	else result.push(b[bi++]);
	return result;
}
function makeJaxpr$1(f, opts) {
	return (...argsIn) => {
		try {
			var _usingCtx$3 = _usingCtx();
			const staticArgnums = new Set(opts?.staticArgnums ?? []);
			const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);
			const [avalsIn, inTree] = flatten(shapedArgs);
			const [fFlat, outTree] = flattenFun((...dynamicArgs) => {
				return f(...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums));
			}, inTree);
			const builder = new JaxprBuilder();
			const main = _usingCtx$3.u(newMain(JaxprTrace, builder));
			_usingCtx$3.u(newDynamic(main));
			const trace = new JaxprTrace(main);
			const tracersIn = avalsIn.map((aval) => trace.newArg(typeof aval === "object" ? aval : pureArray(aval)));
			const tracersOut = fFlat(...tracersIn).map((out) => fullRaise(trace, out));
			const jaxpr = builder.build(tracersIn, tracersOut);
			if (outTree.value === void 0) throw new Error("outTree was not set in makeJaxpr");
			return {
				jaxpr: jaxpr.mapJaxpr((j) => j.simplify()),
				treedef: outTree.value
			};
		} catch (_) {
			_usingCtx$3.e = _;
		} finally {
			_usingCtx$3.d();
		}
	};
}
function jit$1(f, opts) {
	const cache = /* @__PURE__ */ new Map();
	const staticArgnums = new Set(opts?.staticArgnums ?? []);
	const result = ((...args) => {
		const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);
		const [argsFlat, inTree] = flatten(dynamicArgs);
		const avalsIn = unflatten(inTree, argsFlat.map((x) => ShapedArray.fromAval(getAval(x))));
		const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
		const { jaxpr, treedef: outTree } = require_backend.runWithCache(cache, jaxprArgs, () => makeJaxpr$1(f, opts)(...jaxprArgs));
		return unflatten(outTree, bind("jit", [...jaxpr.consts.map((c) => c.ref), ...argsFlat], {
			name: f.name || "closure",
			jaxpr: jaxpr.jaxpr,
			numConsts: jaxpr.consts.length
		}));
	});
	result.dispose = () => {
		for (const { jaxpr } of cache.values()) jaxpr.dispose();
	};
	return result;
}
//#endregion
//#region src/frontend/jit.ts
/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
var JitProgram = class {
	backend;
	steps;
	inputs;
	outputs;
	constructor(backend, steps, inputs, outputs) {
		this.backend = backend;
		this.steps = steps;
		this.inputs = inputs;
		this.outputs = outputs;
	}
	pprint() {
		const steps = this.steps.map((step) => {
			switch (step.type) {
				case "execute": {
					const executeText = `execute (${step.inputs.map((id, i) => `${i}: %${id}`).join(", ")}) -> ${step.outputs.map((id) => `%${id}`).join(", ")}`;
					if (step.source instanceof require_backend.Kernel) return require_backend.PPrint.pp(`${executeText}, kernel`).concat(step.source.pprint().indent(2));
					else if (step.source instanceof require_backend.Routine) return require_backend.PPrint.pp(`${executeText}, routine ${step.source.name}`);
					else {
						step.source;
						return require_backend.PPrint.pp(executeText);
					}
				}
				case "malloc": return require_backend.PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
				case "incref": return require_backend.PPrint.pp(`incref ${step.input}`);
				case "free": return require_backend.PPrint.pp(`free ${step.input}`);
			}
		});
		const display = require_backend.PPrint.prototype.concat(require_backend.PPrint.pp(`device = ${this.backend.type}`), require_backend.PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"), require_backend.PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"), require_backend.PPrint.pp("steps ="), require_backend.PPrint.prototype.concat(...steps).indent(2));
		return require_backend.PPrint.pp("{ ").stack(display.stack(require_backend.PPrint.pp(" }")));
	}
	toString() {
		return this.pprint().toString();
	}
	/** Execute the JitProgram with the given inputs. */
	execute(inputs) {
		const scope = /* @__PURE__ */ new Map();
		if (inputs.length !== this.inputs.length) throw new TypeError(`Expected ${this.inputs.length} inputs, got ${inputs.length}`);
		for (const [i, id] of this.inputs.entries()) scope.set(id, inputs[i]);
		const pending = [];
		for (const step of this.steps) switch (step.type) {
			case "execute": {
				const inputs = step.inputs.map((id) => scope.get(id));
				const outputs = step.outputs.map((id) => scope.get(id));
				if (inputs.some((s) => s === void 0) || outputs.some((s) => s === void 0)) throw new Error(`internal: JitProgram scope undefined`);
				pending.push(new PendingExecute(this.backend, step.source, inputs, outputs));
				break;
			}
			case "malloc": {
				const slot = this.backend.malloc(step.size);
				scope.set(step.output, slot);
				break;
			}
			case "incref": {
				const slot = scope.get(step.input);
				this.backend.incRef(slot);
				break;
			}
			case "free": {
				const slot = scope.get(step.input);
				this.backend.decRef(slot);
				scope.delete(step.input);
				break;
			}
			default:
		}
		return {
			outputs: this.outputs.map((id) => scope.get(id)),
			pending
		};
	}
};
var JitProgramBuilder = class {
	backend;
	#nextId;
	steps;
	constructor(backend, nargs) {
		this.backend = backend;
		this.#nextId = nargs;
		this.steps = [];
	}
	pushLit(lit) {
		const kernel = new require_backend.Kernel(0, lit.aval.size, require_backend.AluExp.const(lit.dtype, lit.value));
		return this.pushKernel(kernel, []);
	}
	pushBuffer(size) {
		const id = this.#nextId++;
		this.steps.push({
			type: "malloc",
			size,
			output: id
		});
		return id;
	}
	pushKernel(kernel, inputs) {
		const id = this.pushBuffer(kernel.bytes);
		this.steps.push({
			type: "execute",
			source: kernel,
			inputs,
			outputs: [id]
		});
		return id;
	}
	pushRoutine(routine, inputs, outputs) {
		this.steps.push({
			type: "execute",
			source: routine,
			inputs,
			outputs
		});
	}
	pushIncref(id) {
		this.steps.push({
			type: "incref",
			input: id
		});
	}
	insertFreeSteps(outputIds) {
		const ids = this.steps.filter((s) => s.type === "malloc").map((s) => s.output);
		for (const id of ids) {
			if (outputIds.includes(id)) continue;
			const lastUsage = this.steps.findLastIndex((s) => s.type === "execute" && (s.outputs.includes(id) || s.inputs.includes(id)) || s.type === "malloc" && s.output === id);
			this.steps.splice(lastUsage + 1, 0, {
				type: "free",
				input: id
			});
		}
	}
	pushFree(id) {
		this.steps.push({
			type: "free",
			input: id
		});
	}
};
const jitCompileCache = /* @__PURE__ */ new Map();
function jitCompile(backend, jaxpr) {
	const cacheKey = backend.type + "," + require_backend.FpHash.hash(jaxpr);
	const cached = jitCompileCache.get(cacheKey);
	if (cached) return cached;
	if (require_backend.DEBUG >= 1) console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
	jaxpr = jaxpr.flatten().simplify();
	const nargs = jaxpr.inBinders.length;
	const builder = new JitProgramBuilder(backend, nargs);
	const blackNodes = splitGraphDataflow(backend, jaxpr);
	const ctx = /* @__PURE__ */ new Map();
	for (let i = 0; i < nargs; i++) {
		const v = jaxpr.inBinders[i];
		ctx.set(v, {
			type: "imm",
			arg: i
		});
	}
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (routinePrimitives.has(eqn.primitive)) {
			const routine = new require_backend.Routine(routinePrimitives.get(eqn.primitive), {
				inputShapes: eqn.inputs.map((x) => x.aval.shape),
				inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
				outputShapes: eqn.outBinders.map((x) => x.aval.shape),
				outputDtypes: eqn.outBinders.map((x) => x.aval.dtype)
			}, eqn.params);
			const inputs = [];
			for (const input of eqn.inputs) if (input instanceof Var) {
				const jv = ctx.get(input);
				if (jv.type !== "imm") throw new Error(`jit: routine primitive ${eqn.primitive} input is not imm`);
				inputs.push(jv.arg);
			} else if (input instanceof Lit) inputs.push(builder.pushLit(input));
			const outputs = [];
			for (const outVar of eqn.outBinders) {
				const outId = builder.pushBuffer(outVar.aval.size * require_backend.byteWidth(outVar.aval.dtype));
				outputs.push(outId);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			}
			builder.pushRoutine(routine, inputs, outputs);
			continue;
		}
		const inputExps = [];
		const inputAvals = [];
		const inputArgs = [];
		let inputReduction = null;
		const addArgs = (args) => {
			const newGids = [];
			for (const jitId of args) {
				let newGid = inputArgs.indexOf(jitId);
				if (newGid === -1) {
					newGid = inputArgs.length;
					inputArgs.push(jitId);
				}
				newGids.push(newGid);
			}
			return newGids;
		};
		for (const input of eqn.inputs) if (input instanceof Var) {
			const jv = ctx.get(input);
			if (jv.type === "exp") {
				const newGids = addArgs(jv.args);
				inputExps.push(jv.exp.reindexGids(newGids));
			} else if (jv.type === "imm") {
				const [gid] = addArgs([jv.arg]);
				const st = require_backend.ShapeTracker.fromShape(input.aval.shape);
				const indices = require_backend.unravelAlu(st.shape, require_backend.AluVar.gidx);
				inputExps.push(require_backend.AluExp.globalView(input.aval.dtype, gid, st, indices));
			} else if (jv.type === "red") {
				if (inputReduction) throw new Error("jit: unexpected, multiple red inputs");
				const newGids = addArgs(jv.args);
				inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
				inputReduction = jv;
			}
			inputAvals.push(input.aval);
		} else if (input instanceof Lit) {
			inputExps.push(require_backend.AluExp.const(input.dtype, input.value));
			inputAvals.push(input.aval);
		} else throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
		const rule = jitRules[eqn.primitive];
		if (!rule) throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
		let exp;
		let reduction;
		if (inputReduction) {
			const jv = inputReduction;
			const newEpilogue = rule(inputExps, inputAvals, eqn.params).exp[0];
			exp = [jv.exp.reindexGids(addArgs(jv.args))];
			reduction = new require_backend.Reduction(jv.reduction.dtype, jv.reduction.op, jv.reduction.size, newEpilogue);
		} else {
			const ruleOutput = rule(inputExps, inputAvals, eqn.params);
			exp = ruleOutput.exp;
			reduction = ruleOutput.reduction;
		}
		for (let i = 0; i < eqn.outBinders.length; i++) {
			const outVar = eqn.outBinders[i];
			if (blackNodes.has(outVar)) {
				const nargs = inputArgs.length;
				const size = outVar.aval.size;
				const kernel = new require_backend.Kernel(nargs, size, exp[i], reduction);
				const outId = builder.pushKernel(kernel, inputArgs);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			} else if (reduction) ctx.set(outVar, {
				type: "red",
				exp: exp[i],
				reduction,
				args: inputArgs
			});
			else ctx.set(outVar, {
				type: "exp",
				exp: exp[i],
				args: inputArgs
			});
		}
	}
	const outputIds = [];
	for (const out of jaxpr.outs) if (out instanceof Var) {
		const jitValue = ctx.get(out);
		if (jitValue.type !== "imm") throw new Error("internal: Expected imm, since outs are black nodes");
		outputIds.push(jitValue.arg);
	} else if (out instanceof Lit) outputIds.push(builder.pushLit(out));
	const outputNeedsRef = new Set(require_backend.range(nargs));
	for (const outputId of outputIds) if (outputNeedsRef.has(outputId)) builder.pushIncref(outputId);
	else outputNeedsRef.add(outputId);
	builder.insertFreeSteps(outputIds);
	const jp = new JitProgram(backend, builder.steps, require_backend.range(0, nargs), outputIds);
	if (require_backend.DEBUG >= 4) console.info(jp.toString());
	jitCompileCache.set(cacheKey, jp);
	return jp;
}
function reshapeViews(exp, mapping, reduceAxis = false) {
	return exp.rewrite((exp) => {
		if (exp.op === "GlobalView") {
			const [gid, st] = exp.arg;
			const newSt = mapping(st);
			if (newSt) {
				const indices = reduceAxis ? require_backend.unravelAlu(newSt.shape.slice(0, -1), require_backend.AluVar.gidx).concat(require_backend.AluVar.ridx) : require_backend.unravelAlu(newSt.shape, require_backend.AluVar.gidx);
				return require_backend.AluExp.globalView(exp.dtype, gid, newSt, indices);
			}
		} else if (exp.op === "GlobalIndex") throw new Error("internal: reshapeViews() called with GlobalIndex op");
	});
}
function broadcastedJit(fn, opts) {
	return (exps, avals, params) => {
		let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);
		const skipCastIdx = opts?.skipCastIdx ?? [];
		if (skipCastIdx.length) newDtype = avals.filter((_, i) => !skipCastIdx.includes(i)).reduce(promoteAvals).dtype;
		exps = exps.map((exp, i) => {
			exp = reshapeViews(exp, (st) => {
				if (!require_backend.deepEqual(st.shape, newShape)) return st.broadcast(newShape, require_backend.range(newShape.length - st.shape.length));
			});
			if (exp.dtype !== newDtype && !skipCastIdx.includes(i)) exp = require_backend.AluExp.cast(newDtype, exp);
			return exp;
		});
		return { exp: [fn(exps, params)] };
	};
}
function unopJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [fn(a, params)] };
	};
}
function reshapeJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [reshapeViews(a, (st) => fn(st, params))] };
	};
}
function routineNoJit() {
	return () => {
		throw new Error("jit: rule is not implemented for routines");
	};
}
const jitRules = {
	["add"]: broadcastedJit(([a, b]) => require_backend.AluExp.add(a, b)),
	["mul"]: broadcastedJit(([a, b]) => require_backend.AluExp.mul(a, b)),
	["idiv"]: broadcastedJit(([a, b]) => require_backend.AluExp.idiv(a, b)),
	["mod"]: broadcastedJit(([a, b]) => require_backend.AluExp.mod(a, b)),
	["min"]: broadcastedJit(([a, b]) => require_backend.AluExp.min(a, b)),
	["max"]: broadcastedJit(([a, b]) => require_backend.AluExp.max(a, b)),
	["bit_combine"]: broadcastedJit(([a, b], { op }) => require_backend.AluExp.bitCombine(a, b, op)),
	["bit_shift"]: broadcastedJit(([a, b], { op }) => require_backend.AluExp.bitShift(a, b, op)),
	["bit_count"]: unopJit(require_backend.AluExp.bitCount),
	["neg"]: unopJit((a) => require_backend.AluExp.sub(require_backend.AluExp.const(a.dtype, 0), a)),
	["reciprocal"]: unopJit(require_backend.AluExp.reciprocal),
	["floor"]: unopJit(require_backend.AluExp.floor),
	["ceil"]: unopJit(require_backend.AluExp.ceil),
	["stop_gradient"]: unopJit((a) => a),
	["cast"]: unopJit((a, { dtype }) => require_backend.AluExp.cast(dtype, a)),
	["bitcast"]: unopJit((a, { dtype }) => require_backend.AluExp.bitcast(dtype, a)),
	["sin"]: unopJit(require_backend.AluExp.sin),
	["cos"]: unopJit(require_backend.AluExp.cos),
	["asin"]: unopJit(require_backend.AluExp.asin),
	["atan"]: unopJit(require_backend.AluExp.atan),
	["exp"]: unopJit(require_backend.AluExp.exp),
	["log"]: unopJit(require_backend.AluExp.log),
	["erf"]: unopJit(require_backend.AluExp.erf),
	["erfc"]: unopJit(require_backend.AluExp.erfc),
	["sqrt"]: unopJit(require_backend.AluExp.sqrt),
	["reduce"]([a], [as], { op, axis }) {
		const keptAxes = [];
		const shiftedAxes = [];
		const newShape = [];
		for (let i = 0; i < as.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(as.shape[i]);
		}
		const reductionSize = require_backend.prod(shiftedAxes.map((ax) => as.shape[ax]));
		newShape.push(reductionSize);
		const perm = keptAxes.concat(shiftedAxes);
		a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
		const reduction = new require_backend.Reduction(a.dtype, op, reductionSize);
		return {
			exp: [a],
			reduction
		};
	},
	["pool"]: reshapeJit((st, { window, strides }) => pool(st, window, strides)),
	["pool_transpose"]([a], [as], { inShape, window, strides }) {
		let stX = poolTranspose(require_backend.ShapeTracker.fromShape(as.shape), inShape, window, strides);
		stX = stX.reshape([...inShape, require_backend.prod(stX.shape.slice(inShape.length))]);
		a = reshapeViews(a, (st) => st.compose(stX), true);
		const reduction = new require_backend.Reduction(a.dtype, "Add", stX.shape[stX.shape.length - 1]);
		return {
			exp: [a],
			reduction
		};
	},
	["dot"]([a, b], [as, bs]) {
		const [c] = jitRules["mul"]([a, b], [as, bs], {}).exp;
		const cs = promoteAvals(as, bs);
		return jitRules["reduce"]([c], [cs], {
			op: "Add",
			axis: [cs.ndim - 1]
		});
	},
	["conv"]([a, b], [as, bs], params) {
		const [stX, stY] = prepareConv(require_backend.ShapeTracker.fromShape(as.shape), require_backend.ShapeTracker.fromShape(bs.shape), params);
		a = reshapeViews(a, (st) => st.compose(stX));
		b = reshapeViews(b, (st) => st.compose(stY));
		as = new ShapedArray(stX.shape, as.dtype, as.weakType);
		bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
		return jitRules["dot"]([a, b], [as, bs], {});
	},
	["compare"]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
	["where"]: broadcastedJit(([cond, a, b]) => require_backend.AluExp.where(cond, a, b), { skipCastIdx: [0] }),
	["concatenate"](exps, avals, { axis }) {
		const ndim = avals[0].ndim;
		const sizes = avals.map((x) => x.shape[axis]);
		const finalSize = sizes.reduce((a, b) => a + b, 0);
		const { dtype: dtypeOut } = avals.map((x) => x.scalar()).reduce(promoteAvals);
		const makePadAxis = (start, end) => require_backend.range(ndim).map((i) => i === axis ? [start, end] : [0, 0]);
		let cum = 0;
		const src = [];
		for (let i = 0; i < exps.length; i++) {
			const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
			src.push(reshapeViews(require_backend.AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)));
			cum += sizes[i];
		}
		return { exp: [src.reduce(require_backend.AluExp.add)] };
	},
	["split"]([a], [as], { axis, sizes }) {
		const exp = [];
		let start = 0;
		for (const size of sizes) {
			const slice = require_backend.range(as.ndim).map((d) => d === axis ? [start, start + size] : [0, as.shape[d]]);
			exp.push(reshapeViews(a, (st) => st.shrink(slice)));
			start += size;
		}
		return { exp };
	},
	["random_bits"]: (keys, keyShapes, { shape, mode }) => {
		const keyShape = keyShapes[0].shape;
		const mapping = (st) => {
			if (!require_backend.deepEqual(st.shape, shape)) return st.broadcast(shape, require_backend.range(st.shape.length, shape.length));
		};
		const k0 = reshapeViews(keys[0], mapping);
		const k1 = reshapeViews(keys[1], mapping);
		const c0 = require_backend.AluExp.u32(0);
		const c1 = require_backend.AluExp.mod(require_backend.AluExp.cast("uint32", require_backend.AluVar.gidx), require_backend.AluExp.u32(Math.max(require_backend.prod(shape.slice(keyShape.length)), 1)));
		return { exp: [require_backend.AluExp.threefry2x32(k0, k1, c0, c1, mode)] };
	},
	["gather"]([x, ...indices], [xs, ...indicesShapes], { axis, outDim }) {
		const axisSet = new Set(axis);
		const indexShape = indicesShapes.map((c) => c.shape).reduce(require_backend.generalBroadcast);
		const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxNonaxis = [...require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx)];
		idxNonaxis.splice(outDim, indexShape.length);
		const src = [...idxNonaxis];
		for (let i = 0; i < xs.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, iexp] of indices.entries()) src[axis[i]] = require_backend.AluExp.cast("int32", reshapeViews(iexp, (st) => st.broadcast(finalShape, [...require_backend.range(outDim + indexShape.length - st.shape.length), ...require_backend.range(outDim + indexShape.length, finalShape.length)])));
		const [index, valid] = require_backend.ShapeTracker.fromShape(xs.shape).toAluExp(src);
		if (!valid.resolve()) throw new Error("internal: expected full validity mask in Gather");
		return { exp: [x.substitute({ gidx: index })] };
	},
	["transpose"]: reshapeJit((st, { perm }) => st.permute(perm)),
	["broadcast"]: reshapeJit((st, { shape, axis }) => st.broadcast(shape, axis)),
	["reshape"]: reshapeJit((st, { shape }) => st.reshape(shape)),
	["flip"]: reshapeJit((st, { axis }) => {
		const arg = require_backend.rep(st.shape.length, false);
		for (const ax of axis) arg[ax] = true;
		return st.flip(arg);
	}),
	["shrink"]: reshapeJit((st, { slice }) => st.shrink(slice)),
	["pad"]: reshapeJit((st, { width }) => st.pad(width)),
	["sort"]: routineNoJit(),
	["argsort"]: routineNoJit(),
	["scatter"]: routineNoJit(),
	["triangular_solve"]: routineNoJit(),
	["cholesky"]: routineNoJit(),
	["lu"]: routineNoJit(),
	["jacobi_eigh"]: routineNoJit(),
	["fft"]: routineNoJit(),
	["jit"]() {
		throw new Error("internal: Jit should have been flattened before JIT compilation");
	}
};
/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend, jaxpr) {
	const varToDefn = /* @__PURE__ */ new Map();
	const varToUsages = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		for (const v of eqn.outBinders) if (v instanceof Var) varToDefn.set(v, i);
		for (const input of eqn.inputs) if (input instanceof Var) {
			const usages = varToUsages.get(input);
			if (usages) usages.push(i);
			else varToUsages.set(input, [i]);
		}
	}
	const reducePrimitives = [
		"reduce",
		"dot",
		"conv",
		"pool_transpose"
	];
	const reductionEpilogueEqns = /* @__PURE__ */ new Set();
	const reductionEndpointEqns = /* @__PURE__ */ new Set();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (reducePrimitives.includes(eqn.primitive)) {
			let head = i;
			while (true) {
				reductionEpilogueEqns.add(head);
				const outVar = jaxpr.eqns[head].outBinders[0];
				const usages = varToUsages.get(outVar) ?? [];
				if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;
				if (reductionEpilogueEqns.has(usages[0])) break;
				const nextEqn = jaxpr.eqns[usages[0]];
				switch (nextEqn.primitive) {
					case "bit_count":
					case "neg":
					case "reciprocal":
					case "floor":
					case "ceil":
					case "stop_gradient":
					case "cast":
					case "bitcast":
					case "sin":
					case "cos":
					case "asin":
					case "atan":
					case "exp":
					case "log":
					case "erf":
					case "erfc":
					case "sqrt":
						head = usages[0];
						continue;
					case "add":
					case "mul":
					case "idiv":
					case "mod":
					case "min":
					case "max":
					case "bit_combine":
					case "bit_shift": {
						const otherInput = nextEqn.inputs.find((v) => v !== outVar);
						if (otherInput instanceof Lit || require_backend.deepEqual(require_backend.generalBroadcast(otherInput.aval.shape, outVar.aval.shape), outVar.aval.shape)) {
							head = usages[0];
							continue;
						}
						break;
					}
				}
				break;
			}
			reductionEndpointEqns.add(head);
		}
	}
	const viewPrimitives = [
		"transpose",
		"broadcast",
		"reshape",
		"flip",
		"shrink",
		"pad"
	];
	const materializedProducers = /* @__PURE__ */ new Set();
	const reductionInputHasReuse = (shape, promoted) => {
		const offset = promoted.length - shape.length;
		for (let axis = 0; axis < promoted.length - 1; axis++) if ((axis < offset ? 1 : shape[axis - offset]) === 1 && promoted[axis] > 1) return true;
		return false;
	};
	const markNonViewProducer = (v) => {
		let current = v;
		while (true) {
			const defn = varToDefn.get(current);
			if (defn === void 0) return;
			const eqn = jaxpr.eqns[defn];
			if (!viewPrimitives.includes(eqn.primitive)) break;
			const input = eqn.inputs.find((input) => input instanceof Var);
			if (!input) return;
			current = input;
		}
		materializedProducers.add(current);
	};
	for (const eqn of jaxpr.eqns) {
		let inputShapes = null;
		if (eqn.primitive === "dot") inputShapes = [eqn.inputs[0].aval.shape, eqn.inputs[1].aval.shape];
		else if (eqn.primitive === "conv") {
			const [lhs, rhs] = prepareConv(require_backend.ShapeTracker.fromShape(eqn.inputs[0].aval.shape), require_backend.ShapeTracker.fromShape(eqn.inputs[1].aval.shape), eqn.params);
			inputShapes = [lhs.shape, rhs.shape];
		}
		if (!inputShapes) continue;
		const promoted = require_backend.generalBroadcast(inputShapes[0], inputShapes[1]);
		for (const [i, input] of eqn.inputs.entries()) if (input instanceof Var && reductionInputHasReuse(inputShapes[i], promoted)) markNonViewProducer(input);
	}
	const blackNodes = /* @__PURE__ */ new Set();
	const p1NextBlack = /* @__PURE__ */ new Map();
	for (const v of jaxpr.outs) if (v instanceof Var) {
		blackNodes.add(v);
		p1NextBlack.set(v, v);
	}
	const heterogeneousViewPrimitives = ["random_bits", "gather"];
	const needsCleanShapePrimitives = ["concatenate", "pad"];
	for (const eqn of jaxpr.eqns) if (needsCleanShapePrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive)) {
		for (const input of eqn.inputs) if (input instanceof Var) markNonViewProducer(input);
	}
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		if (reductionEndpointEqns.has(i) || heterogeneousViewPrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive) || eqn.outBinders.some((v) => materializedProducers.has(v)) || eqn.outBinders.some((v) => blackNodes.has(v))) {
			for (const v of eqn.outBinders) {
				blackNodes.add(v);
				p1NextBlack.set(v, v);
			}
			continue;
		}
		const reach = /* @__PURE__ */ new Set();
		for (const v of eqn.outBinders) for (const j of varToUsages.get(v) ?? []) for (const o of jaxpr.eqns[j].outBinders) {
			const u = p1NextBlack.get(o);
			if (u) reach.add(u);
		}
		if (reach.size > 1) for (const v of eqn.outBinders) {
			blackNodes.add(v);
			p1NextBlack.set(v, v);
		}
		else if (reach.size === 1) {
			const b = reach.values().next().value;
			for (const v of eqn.outBinders) p1NextBlack.set(v, b);
		}
	}
	const p2Deps = /* @__PURE__ */ new Map();
	for (const v of jaxpr.inBinders) p2Deps.set(v, /* @__PURE__ */ new Set([v]));
	let p2idx = 0;
	while (p2idx < jaxpr.eqns.length) {
		const eqn = jaxpr.eqns[p2idx++];
		const deps = [];
		for (const input of eqn.inputs) if (input instanceof Var) if (blackNodes.has(input)) deps.push(/* @__PURE__ */ new Set([input]));
		else deps.push(p2Deps.get(input));
		else deps.push(/* @__PURE__ */ new Set());
		const depCounter = /* @__PURE__ */ new Map();
		for (const depSet of deps) for (const dep of depSet) depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
		if (depCounter.size > backend.maxArgs) {
			let maxUniqueDeps = 0;
			let assocInput = -1;
			for (let i = 0; i < eqn.inputs.length; i++) {
				const input = eqn.inputs[i];
				if (input instanceof Var && varToDefn.has(input)) {
					let uniqueDeps = 0;
					for (const dep of deps[i]) if (depCounter.get(dep) === 1) uniqueDeps++;
					if (uniqueDeps > maxUniqueDeps) {
						maxUniqueDeps = uniqueDeps;
						assocInput = i;
					}
				}
			}
			if (assocInput === -1) throw new Error(`internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`);
			const assocVar = eqn.inputs[assocInput];
			p2idx = varToDefn.get(assocVar);
			for (const out of jaxpr.eqns[p2idx++].outBinders) blackNodes.add(out);
		} else {
			const s = new Set(depCounter.keys());
			for (const out of eqn.outBinders) p2Deps.set(out, s);
		}
	}
	return blackNodes;
}
//#endregion
//#region src/frontend/array.ts
const JsArray$3 = globalThis.Array;
const inlineArrayLimit = 128;
/** Version of pureArray with fudged types. */
const fudgeArray = pureArray;
/**
* An executable operation that will be dispatched to the backend.
*
* This holds a reference to all input buffers used in the operation. After the
* operation is dispatched, the references should be released.
*/
var PendingExecute = class {
	backend;
	source;
	inputs;
	outputs;
	prepared = null;
	submitted = false;
	#promise = null;
	#rc = 1;
	constructor(backend, source, inputs, outputs) {
		this.backend = backend;
		this.source = source;
		this.inputs = inputs;
		this.outputs = outputs;
		for (const slot of inputs) this.backend.incRef(slot);
		for (const slot of outputs) this.backend.incRef(slot);
	}
	updateRc(delta) {
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		this.#rc += delta;
		if (this.#rc <= 0 && !this.submitted) {
			for (const slot of this.inputs) this.backend.decRef(slot);
			for (const slot of this.outputs) this.backend.decRef(slot);
		}
	}
	async prepare() {
		if (this.prepared) return;
		if (this.#promise) {
			await this.#promise;
			return;
		}
		this.#promise = (async () => {
			if (this.source instanceof require_backend.Kernel) this.prepared = await this.backend.prepareKernel(this.source);
			else this.prepared = await this.backend.prepareRoutine(this.source);
		})();
		await this.#promise;
	}
	prepareSync() {
		if (this.prepared) return;
		if (this.source instanceof require_backend.Kernel) this.prepared = this.backend.prepareKernelSync(this.source);
		else this.prepared = this.backend.prepareRoutineSync(this.source);
	}
	submit() {
		if (this.submitted) return;
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		if (!this.prepared) throw new Error("internal: Not prepared yet");
		this.submitted = true;
		this.backend.dispatch(this.prepared, this.inputs, this.outputs);
		for (const slot of this.inputs) this.backend.decRef(slot);
		for (const slot of this.outputs) this.backend.decRef(slot);
	}
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
var Array$1 = class Array$1 extends Tracer {
	#dtype;
	#weakType;
	#source;
	#st;
	#backend;
	#committed;
	#rc;
	#pendingSet;
	/**
	* @ignore
	* Constructs an array from source, shape and backend. Note that if the source
	* is a backend `Slot`, this constructor _takes ownership_ of the slot. It
	* will be freed when the array is disposed.
	*/
	constructor(args) {
		super(baseArrayTrace);
		this.#dtype = args.dtype;
		this.#weakType = args.weakType;
		this.#source = args.source;
		this.#st = args.st;
		this.#backend = args.backend;
		this.#committed = args.committed;
		this.#rc = 1;
		this.#pendingSet = new Set(args.pending);
		if (this.#pendingSet.size === 0) this.#pendingSet = null;
		else if (this.#source instanceof require_backend.AluExp) throw new Error("internal: AluExp source cannot have pending executes");
	}
	/** @ignore */
	get aval() {
		return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
	}
	/** Return a simple string representation of the array's dimensions. */
	toString() {
		return `Array:${this.#dtype}[${this.shape.join(",")}]`;
	}
	get device() {
		return this.#backend.type;
	}
	#check() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
	}
	/** Construct an array, copying fields from `this`. */
	#newArrayFrom(args) {
		return new Array$1({
			source: args.source ?? this.#source,
			st: args.st ?? this.#st,
			dtype: args.dtype ?? this.#dtype,
			weakType: this.#weakType,
			backend: args.backend ?? this.#backend,
			committed: args.committed ?? this.#committed,
			pending: args.pending ?? this.#pending ?? void 0
		});
	}
	get ref() {
		this.#check();
		this.#rc++;
		return this;
	}
	/** Get the current reference count (for debugging memory management). */
	get refCount() {
		return this.#rc;
	}
	dispose() {
		this.#check();
		if (--this.#rc === 0) {
			for (const exe of this.#pending) exe.updateRc(-1);
			if (typeof this.#source === "number") this.#backend.decRef(this.#source);
		}
	}
	/** Get the pending executes as a list, trimming if already submitted. */
	get #pending() {
		if (!this.#pendingSet) return [];
		for (const p of this.#pendingSet) if (p.submitted) this.#pendingSet.delete(p);
		if (this.#pendingSet.size === 0) {
			this.#pendingSet = null;
			return [];
		} else return [...this.#pendingSet];
	}
	/**
	* Convert this array into a primitive value.
	*
	* This only works for scalars (0-dimensional arrays). It lets you get values
	* "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
	* evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
	*
	* This method is also called for `==` equality.
	*/
	[Symbol.toPrimitive]() {
		if (this.ndim === 0) return this.dataSync()[0];
		else throw new Error(`Cannot convert non-scalar array to primitive: ${this.toString()}`);
	}
	#reshape(st) {
		this.#check();
		const pending = this.#pending;
		for (const exe of pending) exe.updateRc(1);
		if (typeof this.#source === "number") this.#backend.incRef(this.#source);
		const ar = this.#newArrayFrom({
			st,
			pending
		});
		this.dispose();
		return ar;
	}
	/**
	* Underlying implementation of the Gather primitive. This indexes an array
	* and extracts slices based on indices in other integer arrays.
	*/
	#gather(indices, axis, outDim) {
		this.#check();
		const axisSet = new Set(axis);
		if (axisSet.size !== axis.length) throw new TypeError("Gather axis must not have duplicates");
		if (indices.some((a) => a.#committed && a.#backend !== this.#backend)) throw new TypeError(`Gather indices must have the same backend: ${this.#backend.type}`);
		indices = indices.map((ar) => ar._putSync(this.#backend));
		indices = Array$1.#broadcastArrays(indices);
		const indexShape = indices[0].shape;
		const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxNonaxis = [...require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx)];
		const idxAxis = idxNonaxis.splice(outDim, indexShape.length);
		const inputs = [];
		const src = [...idxNonaxis];
		for (let i = 0; i < this.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, ar] of indices.entries()) if (ar.#source instanceof require_backend.AluExp) src[axis[i]] = require_backend.AluExp.cast("int32", require_backend.accessorAluExp(ar.#source, ar.#st, idxAxis));
		else {
			let gid = inputs.indexOf(ar.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(ar.#source);
			}
			src[axis[i]] = require_backend.AluExp.cast("int32", require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis));
		}
		let exp;
		if (this.#source instanceof require_backend.AluExp) exp = require_backend.accessorAluExp(this.#source, this.#st, src);
		else {
			let gid = inputs.indexOf(this.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(this.#source);
			}
			exp = require_backend.accessorGlobal(this.#dtype, gid, this.#st, src);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(finalShape), exp);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		this.dispose();
		for (const ar of indices) ar.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(finalShape),
			pending
		});
	}
	/** Move axes to the rightmost dimension of the shape. */
	#moveAxesDown(axis) {
		this.#check();
		if (axis.length === 0) return this.reshape(this.shape.concat(1));
		const newShape = [];
		const keptAxes = [];
		const shiftedAxes = [];
		for (let i = 0; i < this.#st.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(this.#st.shape[i]);
		}
		const collapsedAxisSize = require_backend.prod(shiftedAxes.map((axis) => this.#st.shape[axis]));
		newShape.push(collapsedAxisSize);
		return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
	}
	#transpose(perm) {
		this.#check();
		if (!require_backend.isPermutation(perm, this.ndim)) throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
		return this.#reshape(this.#st.permute(perm));
	}
	#unary(op, dtypeOutput) {
		const weakType = !dtypeOutput && this.#weakType;
		dtypeOutput ??= this.#dtype;
		this.#check();
		if (this.#source instanceof require_backend.AluExp) {
			const exp = new require_backend.AluExp(op, dtypeOutput, [this.#source]);
			this.dispose();
			return this.#newArrayFrom({
				source: exp.simplify(),
				dtype: dtypeOutput,
				weakType
			});
		}
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		const exp = new require_backend.AluExp(op, dtypeOutput, [require_backend.AluExp.globalView(this.#dtype, 0, this.#st, indices)]);
		const kernel = new require_backend.Kernel(1, this.#st.size, exp);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, [this.#source], [output]));
		this.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(this.shape),
			dtype: dtypeOutput,
			weakType,
			pending
		});
	}
	#binary(op, other) {
		const custom = (src) => new require_backend.AluExp(op, src[0].dtype, src);
		return Array$1.#naryCustom(op, custom, [this, other]);
	}
	static #naryCustom(name, custom, arrays, { dtypeOverride, strongTypeOutput, reduceAxis } = {}) {
		const n = arrays.length;
		if (n === 0) throw new TypeError(`No inputs for ${name}`);
		for (const ar of arrays) ar.#check();
		let castDtype;
		let castWeakType = true;
		for (let i = 0; i < n; i++) if (dtypeOverride?.[i]) {
			if (arrays[i].#dtype !== dtypeOverride[i]) throw new TypeError(`Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`);
		} else if (castDtype === void 0) {
			castDtype = arrays[i].#dtype;
			castWeakType = arrays[i].#weakType;
		} else ({dtype: castDtype, weakType: castWeakType} = promoteAvals(new ShapedArray([], castDtype, castWeakType), arrays[i].aval.scalar()));
		const weakType = castWeakType && !strongTypeOutput;
		const { backend, committed } = Array$1.#computeBackend(name, arrays);
		arrays = arrays.map((ar) => ar._putSync(backend));
		arrays = Array$1.#broadcastArrays(arrays);
		const newShape = [...arrays[0].shape];
		if (arrays.every((ar) => ar.#source instanceof require_backend.AluExp) && !reduceAxis) {
			const sources = arrays.map((ar, i) => {
				if (!dtypeOverride?.[i]) return require_backend.AluExp.cast(castDtype, ar.#source);
				else return ar.#source;
			});
			if (arrays.every((ar) => require_backend.deepEqual(ar.#st, arrays[0].#st))) {
				const exp = custom(sources);
				arrays.forEach((ar) => ar.dispose());
				return new Array$1({
					source: exp.simplify(),
					st: arrays[0].#st,
					dtype: exp.dtype,
					weakType,
					backend,
					committed
				});
			}
			const exp = custom(arrays.map((ar, i) => {
				const src = sources[i];
				if (ar.#st.contiguous) return src;
				return require_backend.accessorAluExp(src, ar.#st, require_backend.unravelAlu(newShape, require_backend.AluVar.idx));
			}));
			const st = require_backend.ShapeTracker.fromShape(newShape);
			arrays.forEach((ar) => ar.dispose());
			return new Array$1({
				source: exp.simplify(),
				st,
				dtype: exp.dtype,
				weakType,
				backend,
				committed
			});
		}
		let indices;
		if (!reduceAxis) indices = require_backend.unravelAlu(newShape, require_backend.AluVar.gidx);
		else indices = [...require_backend.unravelAlu(newShape.slice(0, -1), require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		const inputs = [];
		const src = [];
		for (const [i, ar] of arrays.entries()) {
			let nextSrc;
			if (ar.#source instanceof require_backend.AluExp) nextSrc = require_backend.accessorAluExp(ar.#source, ar.#st, indices);
			else {
				let gid = inputs.indexOf(ar.#source);
				if (gid === -1) {
					gid = inputs.length;
					inputs.push(ar.#source);
				}
				nextSrc = require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
			}
			if (!dtypeOverride?.[i]) nextSrc = require_backend.AluExp.cast(castDtype, nextSrc);
			src.push(nextSrc);
		}
		const exp = custom(src);
		let re = void 0;
		if (reduceAxis) {
			const [axisSize] = newShape.splice(-1, 1);
			re = new require_backend.Reduction(exp.dtype, "Add", axisSize);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(newShape), exp, re);
		const output = backend.malloc(kernel.bytes);
		const pending = /* @__PURE__ */ new Set([...arrays.flatMap((ar) => ar.#pending)]);
		for (const exe of pending) exe.updateRc(1);
		pending.add(new PendingExecute(backend, kernel, inputs, [output]));
		arrays.forEach((ar) => ar.dispose());
		return new Array$1({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			dtype: kernel.dtype,
			weakType,
			backend,
			committed,
			pending
		});
	}
	/** Reduce the last dimension of the array by an operation. */
	#reduce(op) {
		const shape = this.shape;
		const reduction = new require_backend.Reduction(this.#dtype, op, shape[shape.length - 1]);
		const newShape = shape.slice(0, -1);
		const newSize = require_backend.prod(newShape);
		const indices = [...require_backend.unravelAlu(newShape, require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		let exp;
		const inputs = [];
		if (this.#source instanceof require_backend.AluExp) exp = require_backend.accessorAluExp(this.#source, this.#st, indices);
		else {
			inputs.push(this.#source);
			exp = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
		}
		const kernel = new require_backend.Kernel(inputs.length, newSize, exp, reduction);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		this.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			pending
		});
	}
	/** Apply an operation with custom lowering to this array. */
	static #routine(prim) {
		return (arrays, params) => {
			const { backend, committed } = Array$1.#computeBackend(prim, arrays);
			for (const ar of arrays) ar.#realize();
			const avals = arrays.map((ar) => ar.aval);
			const avalsOut = abstractEvalRules[prim](avals, params);
			const routine = new require_backend.Routine(routinePrimitives.get(prim), {
				inputShapes: avals.map((a) => a.shape),
				inputDtypes: avals.map((a) => a.dtype),
				outputShapes: avalsOut.map((a) => a.shape),
				outputDtypes: avalsOut.map((a) => a.dtype)
			}, params);
			const inputs = arrays.map((ar) => ar.#source);
			const outputs = avalsOut.map((x) => backend.malloc(require_backend.byteWidth(x.dtype) * x.size));
			const pending = arrays.flatMap((ar) => ar.#pending);
			for (const exe of pending) exe.updateRc(+outputs.length);
			pending.push(new PendingExecute(backend, routine, inputs, outputs));
			pending[pending.length - 1].updateRc(+outputs.length - 1);
			arrays.forEach((ar) => ar.dispose());
			return outputs.map((output, i) => new Array$1({
				source: output,
				st: require_backend.ShapeTracker.fromShape(avalsOut[i].shape),
				dtype: avalsOut[i].dtype,
				weakType: avalsOut[i].weakType,
				backend,
				committed,
				pending
			}));
		};
	}
	/**
	* Normalizes this array into one backed by a `Slot`.
	*
	* This mutates the array in-place, turning it into an equivalent array whose
	* source is actual, contiguous data on device.
	*
	* Calling this twice is a no-op.
	*/
	#realize() {
		this.#check();
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		if (this.#source instanceof require_backend.AluExp) {
			let resolvedSource;
			if (this.#st.contiguous && this.#st.size < inlineArrayLimit && (resolvedSource = this.#source.resolve()) !== void 0) {
				const byteLength = this.#st.size * require_backend.byteWidth(this.#dtype);
				const initialData = new Uint8Array(byteLength);
				require_backend.dtypedArray(this.#dtype, initialData).fill(resolvedSource);
				this.#source = this.#backend.malloc(byteLength, initialData);
				this.#st = require_backend.ShapeTracker.fromShape(this.shape);
				return;
			}
			const exp = require_backend.accessorAluExp(this.#source, this.#st, indices);
			const kernel = new require_backend.Kernel(0, this.#st.size, exp);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [], [output]);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet = /* @__PURE__ */ new Set([pendingItem]);
		} else {
			if (this.#st.contiguous) return;
			const exp = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
			const kernel = new require_backend.Kernel(1, this.#st.size, exp);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [this.#source], [output]);
			this.#backend.decRef(this.#source);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet ??= /* @__PURE__ */ new Set();
			this.#pendingSet.add(pendingItem);
		}
	}
	#dataInline() {
		this.#check();
		if (!(this.#source instanceof require_backend.AluExp)) throw new Error("internal: #dataInline called on non-AluExp source");
		const ar = this.#newArrayFrom({ backend: require_backend.getBackend("cpu") });
		this.dispose();
		return ar.dataSync();
	}
	static #broadcastArrays(arrays) {
		if (arrays.length === 0) throw new Error("Need at least one array to broadcast");
		if (arrays.length === 1) return arrays;
		const newShape = arrays.map((a) => a.shape).reduce(require_backend.generalBroadcast);
		return arrays.map((ar) => {
			if (require_backend.deepEqual(ar.shape, newShape)) return ar;
			return ar.#reshape(ar.#st.broadcast(newShape, require_backend.range(newShape.length - ar.ndim)));
		});
	}
	static #computeBackend(name, arrays) {
		const committed = arrays.filter((ar) => ar.#committed);
		if (committed.length > 0) {
			const backend = committed[0].#backend;
			for (const ar of committed) if (ar.#backend !== backend) throw new Error(`Device mismatch in ${name} between committed arrays on (${backend.type}, ${ar.#backend.type}), please move to the same device with devicePut()`);
			return {
				backend,
				committed: true
			};
		} else return {
			backend: arrays.length > 0 ? arrays[0].#backend : require_backend.getBackend(),
			committed: false
		};
	}
	/** Realize the array and return it as data. */
	async data() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = await this.#backend.read(this.#source, 0, byteCount);
		this.dispose();
		return require_backend.dtypedArray(this.dtype, buf);
	}
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
	async blockUntilReady() {
		this.#check();
		if (this.#source instanceof require_backend.AluExp) return this;
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		await this.#backend.read(this.#source, 0, 0);
		return this;
	}
	/**
	* Realize the array and return it as data. This is a sync variant and not
	* recommended for performance reasons, as it will block rendering.
	*/
	dataSync() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		for (const p of this.#pending) {
			p.prepareSync();
			p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = this.#backend.readSync(this.#source, 0, byteCount);
		this.dispose();
		return require_backend.dtypedArray(this.dtype, buf);
	}
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
	async gpuBuffer() {
		if (this.device !== "webgpu") throw new Error(`gpuBuffer() is only available on WebGPU backend`);
		this.#realize();
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		const { buffer } = this.#backend.buffers.get(this.#source);
		this.dispose();
		return buffer;
	}
	/** Synchronous version of `Array.gpuBuffer()`. */
	gpuBufferSync() {
		if (this.device !== "webgpu") throw new Error(`gpuBufferSync() is only available on WebGPU backend`);
		this.#realize();
		for (const p of this.#pending) {
			p.prepareSync();
			p.submit();
		}
		const { buffer } = this.#backend.buffers.get(this.#source);
		this.dispose();
		return buffer;
	}
	/**
	* Convert this array into a JavaScript object.
	*
	* This is a blocking operation that will compile all of the shaders and wait
	* for execution to complete, synchronously. No other JavaScript code on the
	* site will be run during shader execution.
	*
	* To avoid blocking, prefer `jsAsync()` when possible.
	*/
	js() {
		return dataToJs(this.dtype, this.dataSync(), this.shape);
	}
	/** Convert this array into a JavaScript object, asynchronously. */
	async jsAsync() {
		return dataToJs(this.dtype, await this.data(), this.shape);
	}
	/**
	* Copy an element of an array to a numeric scalar and return it.
	*
	* Throws an error if the array does not have a single element. The array must
	* either be rank-0, or all dimensions of the shape are 1.
	*/
	item() {
		if (this.size !== 1) throw new Error(`item() can only be called on arrays of size 1`);
		return this.dataSync()[0];
	}
	/** @private Internal plumbing method for Array / Tracer ops. */
	static _implRules() {
		return {
			["add"]([x, y]) {
				return [x.#binary("Add", y)];
			},
			["mul"]([x, y]) {
				return [x.#binary("Mul", y)];
			},
			["idiv"]([x, y]) {
				return [x.#binary("Idiv", y)];
			},
			["mod"]([x, y]) {
				return [x.#binary("Mod", y)];
			},
			["min"]([x, y]) {
				return [x.#binary("Min", y)];
			},
			["max"]([x, y]) {
				return [x.#binary("Max", y)];
			},
			["bit_combine"]([x, y], { op }) {
				const custom = (src) => require_backend.AluExp.bitCombine(src[0], src[1], op);
				return [Array$1.#naryCustom("bit_combine", custom, [x, y])];
			},
			["bit_shift"]([x, y], { op }) {
				const custom = (src) => require_backend.AluExp.bitShift(src[0], src[1], op);
				return [Array$1.#naryCustom("bit_shift", custom, [x, y], { dtypeOverride: [void 0, y.dtype] })];
			},
			["bit_count"]([x]) {
				return [x.#unary("BitCount", "int32")];
			},
			["neg"]([x]) {
				return [zerosLike$1(x.ref).#binary("Sub", x)];
			},
			["reciprocal"]([x]) {
				return [x.#unary("Reciprocal")];
			},
			["floor"]([x]) {
				return [x.#unary("Floor")];
			},
			["ceil"]([x]) {
				return [x.#unary("Ceil")];
			},
			["stop_gradient"]([x]) {
				return [x];
			},
			["cast"]([x], { dtype }) {
				return [x.#unary("Cast", dtype)];
			},
			["bitcast"]([x], { dtype }) {
				if (x.dtype === dtype) return [x];
				if (x.dtype === "bool" || dtype === "bool") throw new TypeError("Bitcast to/from bool is not allowed");
				if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
				if (x.#source instanceof require_backend.AluExp) return [x.#unary("Bitcast", dtype)];
				else {
					x.#backend.incRef(x.#source);
					const pending = x.#pending;
					for (const exe of pending) exe.updateRc(1);
					const y = x.#newArrayFrom({
						dtype,
						weakType: false,
						pending
					});
					x.dispose();
					return [y];
				}
			},
			["sin"]([x]) {
				return [x.#unary("Sin")];
			},
			["cos"]([x]) {
				return [x.#unary("Cos")];
			},
			["asin"]([x]) {
				return [x.#unary("Asin")];
			},
			["atan"]([x]) {
				return [x.#unary("Atan")];
			},
			["exp"]([x]) {
				return [x.#unary("Exp")];
			},
			["log"]([x]) {
				return [x.#unary("Log")];
			},
			["erf"]([x]) {
				return [x.#unary("Erf")];
			},
			["erfc"]([x]) {
				return [x.#unary("Erfc")];
			},
			["sqrt"]([x]) {
				return [x.#unary("Sqrt")];
			},
			["reduce"]([x], { op, axis }) {
				if (axis.length === 0) return [x];
				return [x.#moveAxesDown(axis).#reduce(op)];
			},
			["pool"]([x], { window, strides }) {
				const st = pool(x.#st, window, strides);
				return [x.#reshape(st)];
			},
			["pool_transpose"]([x], { inShape, window, strides }) {
				const n = inShape.length;
				let st = poolTranspose(x.#st, inShape, window, strides);
				st = st.reshape([...st.shape.slice(0, n), require_backend.prod(st.shape.slice(n))]);
				return [x.#reshape(st).#reduce("Add")];
			},
			["dot"]([x, y]) {
				return [Array$1.#naryCustom("dot", ([x, y]) => require_backend.AluExp.mul(x, y), [x, y], { reduceAxis: true })];
			},
			["conv"]([x, y], params) {
				checkConvShape(x.shape, y.shape, params);
				if (x.#backend.type === "wasm" && params.lhsDilation.every((d) => d === 1) && params.padding.some(([left, right]) => left > 0 || right > 0)) {
					x = x.#reshape(x.#st.padOrShrink([...require_backend.rep(params.vmapDims + 2, [0, 0]), ...params.padding]));
					x.#realize();
					params = {
						...params,
						padding: require_backend.rep(params.padding.length, [0, 0])
					};
				}
				const [stX, stY] = prepareConv(x.#st, y.#st, params);
				return [Array$1.#naryCustom("conv", ([x, y]) => require_backend.AluExp.mul(x, y), [x.#reshape(stX), y.#reshape(stY)], { reduceAxis: true })];
			},
			["compare"]([x, y], { op }) {
				const custom = ([x, y]) => aluCompare(x, y, op);
				return [Array$1.#naryCustom("compare", custom, [x, y], { strongTypeOutput: true })];
			},
			["where"]([cond, x, y]) {
				const custom = ([cond, x, y]) => require_backend.AluExp.where(cond, x, y);
				return [Array$1.#naryCustom("where", custom, [
					cond,
					x,
					y
				], { dtypeOverride: ["bool"] })];
			},
			["concatenate"](xs, { axis }) {
				const ndim = xs[0].ndim;
				const sizes = xs.map((x) => x.shape[axis]);
				const finalSize = sizes.reduce((a, b) => a + b, 0);
				const makePadAxis = (start, end) => require_backend.range(ndim).map((i) => i === axis ? [start, end] : [0, 0]);
				let cum = 0;
				const xsPadded = [];
				for (let i = 0; i < xs.length; i++) {
					const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
					xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
					cum += sizes[i];
				}
				const custom = (exps) => exps.reduce(require_backend.AluExp.add);
				return [Array$1.#naryCustom("concatenate", custom, xsPadded)];
			},
			["split"]([x], { axis, sizes }) {
				const outputs = [];
				for (let i = 0, start = 0; i < sizes.length; i++) {
					const slice = require_backend.range(x.ndim).map((d) => d === axis ? [start, start + sizes[i]] : [0, x.shape[d]]);
					outputs.push(x.ref.#reshape(x.#st.shrink(slice)));
					start += sizes[i];
				}
				x.dispose();
				return outputs;
			},
			["random_bits"]([k0, k1], { shape, mode }) {
				const keyShape = k0.shape;
				const genShape = shape.slice(keyShape.length);
				const c0 = zeros(genShape, {
					dtype: "uint32",
					device: k0.device
				});
				const c1 = arange(0, require_backend.prod(genShape), 1, {
					dtype: "uint32",
					device: k0.device
				}).reshape(genShape);
				k0 = k0.#reshape(k0.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
				k1 = k1.#reshape(k1.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
				const custom = ([k0, k1, c0, c1]) => require_backend.AluExp.threefry2x32(k0, k1, c0, c1, mode);
				return [Array$1.#naryCustom("random_bits", custom, [
					k0,
					k1,
					c0,
					c1
				])];
			},
			["gather"]([x, ...indices], { axis, outDim }) {
				return [x.#gather(indices, axis, outDim)];
			},
			["transpose"]([x], { perm }) {
				return [x.#transpose(perm)];
			},
			["broadcast"]([x], { shape, axis }) {
				return [x.#reshape(x.#st.broadcast(shape, axis))];
			},
			["reshape"]([x], { shape }) {
				return [x.#reshape(x.#st.reshape(shape))];
			},
			["flip"]([x], { axis }) {
				const arg = require_backend.rep(x.ndim, false);
				for (const ax of axis) arg[ax] = true;
				return [x.#reshape(x.#st.flip(arg))];
			},
			["shrink"]([x], { slice }) {
				return [x.#reshape(x.#st.shrink(slice))];
			},
			["pad"]([x], { width }) {
				return [x.#reshape(x.#st.pad(width))];
			},
			["sort"]: Array$1.#routine("sort"),
			["argsort"]: Array$1.#routine("argsort"),
			["scatter"]: Array$1.#routine("scatter"),
			["triangular_solve"]: Array$1.#routine("triangular_solve"),
			["cholesky"]: Array$1.#routine("cholesky"),
			["lu"]: Array$1.#routine("lu"),
			["jacobi_eigh"]: Array$1.#routine("jacobi_eigh"),
			["fft"]: Array$1.#routine("fft"),
			["jit"](args, { jaxpr }) {
				if (jaxpr.inBinders.length !== args.length) throw new Error(`jit expects ${jaxpr.inBinders.length} args, got ${args.length}`);
				const { backend, committed } = Array$1.#computeBackend("jit", args);
				args = args.map((ar) => ar._putSync(backend));
				const { outputs, pending } = jitCompile(backend, jaxpr).execute(args.map((x) => x._realizeSource()));
				for (const exe of pending) exe.updateRc(+outputs.length - 1);
				const prevPending = [...new Set(args.flatMap((x) => x.#pending))];
				for (const exe of prevPending) exe.updateRc(+outputs.length);
				pending.splice(0, 0, ...prevPending);
				args.forEach((x) => x.dispose());
				return outputs.map((source, i) => {
					return new Array$1({
						source,
						st: require_backend.ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
						dtype: jaxpr.outs[i].aval.dtype,
						weakType: jaxpr.outs[i].aval.weakType,
						backend,
						committed,
						pending
					});
				});
			}
		};
	}
	/** @private */
	_realizeSource() {
		this.#realize();
		return this.#source;
	}
	/** @private Put this array on a new backend, asynchronously. */
	async _put(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) {
			const ar = this.#newArrayFrom({
				backend,
				committed: true
			});
			this.dispose();
			return ar;
		} else return arrayFromData(await this.data(), this.shape, {
			dtype: this.#dtype,
			device: backend.type
		}, this.#weakType);
	}
	/** @private Put this array on a new backend, synchronously. */
	_putSync(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) {
			const ar = this.#newArrayFrom({
				backend,
				committed: true
			});
			this.dispose();
			return ar;
		} else return arrayFromData(this.dataSync(), this.shape, {
			dtype: this.#dtype,
			device: backend.type
		}, this.#weakType);
	}
};
/** Constructor for creating a new array from data. */
function array(values, { shape, dtype, device } = {}) {
	if (values instanceof Tracer) {
		if (shape && !require_backend.deepEqual(values.shape, shape)) values = values.reshape(shape);
		if (dtype && values.dtype !== dtype) values = values.astype(dtype);
		return values;
	} else if (ArrayBuffer.isView(values)) return arrayFromData(values, shape ?? [values.length], {
		dtype,
		device
	});
	else {
		if (!shape) {
			shape = [];
			let cur = values;
			while (JsArray$3.isArray(cur)) {
				shape.push(cur.length);
				cur = cur[0];
			}
		}
		const size = require_backend.prod(shape);
		const flat = require_backend.recursiveFlatten(values);
		if (flat.length !== size) throw new Error(`Jagged shape: ${JSON.stringify(shape)} vs ${flat.length}`);
		if (size === 0) return zeros(shape, {
			dtype,
			device
		});
		if (size === 1) return full(shape, flat[0], {
			dtype,
			device
		});
		if (typeof flat[0] === "boolean") {
			dtype = dtype ?? "bool";
			return arrayFromData(new Int32Array(flat.map((x) => x ? 1 : 0)), shape, {
				dtype,
				device
			});
		} else {
			const weakType = dtype == void 0 && shape.length === 0;
			dtype = dtype ?? "float32";
			return arrayFromData(require_backend.dtypedJsArray(dtype, flat), shape, {
				dtype,
				device
			}, weakType);
		}
	}
}
function arrayFromData(data, shape, { dtype, device }, weakType = false) {
	if (data instanceof Float32Array) {
		if (dtype && dtype !== "float32") throw new Error("Float32Array must have float32 type");
		dtype ??= "float32";
	} else if (data instanceof Int32Array) {
		if (dtype && dtype !== "int32" && dtype !== "bool") throw new Error("Int32Array must have int32 or bool type");
		dtype ??= "int32";
	} else if (data instanceof Uint32Array) {
		if (dtype && dtype !== "uint32") throw new Error("Uint32Array must have uint32 type");
		dtype ??= "uint32";
	} else if (data instanceof Float16Array) {
		if (dtype && dtype !== "float16") throw new Error("Float16Array must have float16 type");
		dtype ??= "float16";
	} else if (data instanceof Float64Array) {
		if (dtype && dtype !== "float64") throw new Error("Float64Array must have float64 type");
		dtype ??= "float64";
	} else throw new Error("Unsupported data array type: " + data.constructor.name);
	if (data.length < inlineArrayLimit) {
		let allEqual = true;
		for (let i = 1; i < data.length; i++) if (data[i] !== data[0] || !Object.is(data[i], data[0])) {
			allEqual = false;
			break;
		}
		if (allEqual) return fullInternal(new ShapedArray(shape, dtype, weakType), data[0], device);
	}
	const backend = require_backend.getBackend(device);
	const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	return new Array$1({
		source: backend.malloc(data.byteLength, buf),
		st: require_backend.ShapeTracker.fromShape(shape),
		dtype,
		weakType,
		backend,
		committed: device != void 0
	});
}
function dataToJs(dtype, data, shape) {
	if (shape.length === 0) return dtype === "bool" ? Boolean(data[0]) : data[0];
	const [first, ...rest] = shape;
	const restSize = require_backend.prod(rest);
	const ret = [];
	for (let i = 0; i < first; i++) {
		const subarray = data.slice(i * restSize, (i + 1) * restSize);
		ret.push(dataToJs(dtype, subarray, rest));
	}
	return ret;
}
/** If x is a value, lift it into an array, otherwise leave it be. */
function pureArray(x) {
	if (x instanceof Tracer) return x;
	else return array(x);
}
var EvalTrace = class extends Trace {
	pure = (x) => pureArray(x);
	lift = (x) => x;
	processPrimitive(primitive, tracers, params) {
		return implRules[primitive](tracers, params);
	}
};
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));
const implRules = Array$1._implRules();
function fullInternal(aval, fillValue, device) {
	return new Array$1({
		source: require_backend.AluExp.const(aval.dtype, fillValue),
		st: require_backend.ShapeTracker.fromShape(aval.shape),
		dtype: aval.dtype,
		weakType: aval.weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
function zerosLike$1(val, opts) {
	return fullLike$1(val, 0, opts);
}
function onesLike$1(val, opts) {
	return fullLike$1(val, 1, opts);
}
function fullLike$1(val, fillValue, { dtype, shape, device } = {}) {
	const aval = getAval(val);
	if (val instanceof Tracer) val.dispose();
	if (fillValue instanceof Tracer) throw new Error("numpy.fullLike() with array argument not implemented yet");
	return fullInternal(new ShapedArray(shape ?? aval.shape, dtype ?? aval.dtype, aval.weakType && dtype === void 0), fillValue, device);
}
/** Return a new array of given shape and type, filled with zeros. */
function zeros(shape, opts) {
	return full(shape, 0, opts);
}
/** Return a new array of given shape and type, filled with ones. */
function ones(shape, opts) {
	return full(shape, 1, opts);
}
/** Return a new array of given shape and type, filled with `fill_value`. */
function full(shape, fillValue, { dtype, device } = {}) {
	let weakType = dtype == void 0 && shape.length === 0;
	if (typeof fillValue === "number") dtype = dtype ?? "float32";
	else if (typeof fillValue === "boolean") {
		dtype = dtype ?? "bool";
		weakType = false;
	} else if (fillValue instanceof Tracer) throw new Error("numpy.full() with array argument not implemented yet");
	else throw new TypeError(`Invalid type for full: ${fillValue}`);
	return fullInternal(new ShapedArray(shape, dtype, weakType), fillValue, device);
}
/**
* Create an identity matrix.
*
* If numCols is not provided, it defaults to numRows, i.e., a square identity
* matrix with ones on the diagonal.
*/
function eye(numRows, numCols, { dtype, device } = {}) {
	numCols = numCols ?? numRows;
	const weakType = dtype == void 0;
	dtype = dtype ?? "float32";
	if (numCols < numRows) return eye(numCols, numRows, {
		dtype,
		device
	}).transpose();
	if (numRows === 0) return zeros([0, numCols], {
		dtype,
		device
	});
	const exp = require_backend.AluExp.cmplt(require_backend.AluExp.mod(require_backend.AluVar.idx, require_backend.AluExp.i32(numCols + 1)), require_backend.AluExp.i32(1));
	return new Array$1({
		source: require_backend.AluExp.cast(dtype, exp),
		st: require_backend.ShapeTracker.fromShape([numRows, numCols]),
		dtype,
		weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/** Return the identity matrix, with ones on the main diagonal. */
function identity$1(n, { dtype, device } = {}) {
	return eye(n, n, {
		dtype,
		device
	});
}
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
function arange(start, stop, step = 1, { dtype, device } = {}) {
	dtype = dtype ?? "int32";
	if (stop === void 0) {
		stop = start;
		start = 0;
	}
	if (step === 0) throw new RangeError(`Invalid step for arange: ${step}. Step must be non-zero.`);
	const size = Math.max(0, Math.ceil((stop - start) / step));
	if (size === 0) return zeros([0], {
		dtype,
		device
	});
	return new Array$1({
		source: require_backend.AluExp.add(require_backend.AluExp.const(dtype, start), require_backend.AluExp.mul(require_backend.AluExp.cast(dtype, require_backend.AluVar.idx), require_backend.AluExp.const(dtype, step))),
		st: require_backend.ShapeTracker.fromShape([size]),
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/**
* Return an array with ones on and below the diagonal and zeros elsewhere.
*
* If `k` is provided, it specifies the sub-diagonal on and below which the
* array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
* `k>0` is above it.
*/
function tri(n, m, k = 0, { dtype, device } = {}) {
	m ??= n;
	dtype ??= "float32";
	if (!Number.isInteger(n) || n < 0) throw new Error(`tri: n must be a non-negative integer, got ${n}`);
	if (!Number.isInteger(m) || m < 0) throw new Error(`tri: m must be a non-negative integer, got ${m}`);
	if (!Number.isInteger(k)) throw new Error(`tri: k must be an integer, got ${k}`);
	const rows = arange(k, n + k, 1, {
		dtype: "int32",
		device
	});
	const cols = arange(0, m, 1, {
		dtype: "int32",
		device
	});
	return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}
/** Return the lower triangle of an array. Must be of dimension >= 2. */
function tril(a, k = 0) {
	if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
	a = fudgeArray(a);
	const [n, m] = a.shape.slice(-2);
	return where$1(tri(n, m, k, { dtype: "bool" }), a.ref, zerosLike$1(a));
}
/** Return the upper triangle of an array. Must be of dimension >= 2. */
function triu(a, k = 0) {
	if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
	a = fudgeArray(a);
	const [n, m] = a.shape.slice(-2);
	return where$1(tri(n, m, k - 1, { dtype: "bool" }), zerosLike$1(a.ref), a);
}
/**
* Return evenly spaced numbers over a specified interval.
*
* Returns _num_ evenly spaced samples, calculated over the interval
* [`start`, `stop`]. The endpoint `stop` is included in the result by default,
* but this is controlled by the `endpoint` parameter.
*
* The default data type is Float32. Use arange() for integer steps.
*/
function linspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
	dtype = dtype ?? "float32";
	if (num < 0 || !Number.isInteger(num)) throw new RangeError(`Invalid num for linspace: ${num}. Must be non-negative integer.`);
	else if (num === 0) return zeros([0], {
		dtype,
		device
	});
	else if (num === 1) return full([1], start, {
		dtype,
		device
	});
	else if (start === stop) return full([num], start, {
		dtype,
		device
	});
	const delta = stop - start;
	const denom = endpoint ? num - 1 : num;
	return new Array$1({
		source: require_backend.AluExp.cast(dtype, require_backend.AluExp.add(require_backend.AluExp.f32(start), require_backend.AluExp.mul(require_backend.AluExp.f32(delta / denom), require_backend.AluExp.cast("float32", require_backend.AluVar.idx)))),
		st: require_backend.ShapeTracker.fromShape([num]),
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
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
function logspace(start, stop, num = 50, endpoint = true, base = 10, { dtype, device } = {}) {
	return exp$1(mul(linspace(start, stop, num, endpoint, {
		dtype,
		device
	}), Math.log(base)));
}
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
function geomspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
	if (start === 0 || stop === 0 || Math.sign(start) !== Math.sign(stop)) throw new RangeError(`geomspace: start (${start}) and stop (${stop}) must be nonzero and have the same sign`);
	dtype ??= "float32";
	const computationDtype = require_backend.isFloatDtype(dtype) ? dtype : "float32";
	const sign = Math.sign(start);
	const y = logspace(Math.log10(Math.abs(start)), Math.log10(Math.abs(stop)), num, endpoint, 10, {
		dtype: computationDtype,
		device
	});
	const result = sign < 0 ? y.mul(-1) : y;
	return result.dtype === dtype ? result : result.astype(dtype);
}
function aluCompare(a, b, op) {
	switch (op) {
		case "less": return require_backend.AluExp.cmplt(a, b);
		case "equal": return require_backend.AluExp.cmpne(a, b).not();
		case "not_equal": return require_backend.AluExp.cmpne(a, b);
		case "less_equal": return require_backend.AluExp.add(require_backend.AluExp.cmplt(a, b), require_backend.AluExp.cmpne(a, b).not());
	}
}
//#endregion
//#region src/frontend/vmap.ts
function mappedAval(batchDim, aval) {
	const shape = [...aval.shape];
	shape.splice(batchDim, 1);
	return new ShapedArray(shape, aval.dtype, aval.weakType);
}
/** Move one axis to a different index. */
function moveaxis$1(x, src, dst) {
	const t = pureArray(x);
	src = require_backend.checkAxis(src, t.ndim);
	dst = require_backend.checkAxis(dst, t.ndim);
	if (src === dst) return t;
	const perm = require_backend.range(t.ndim);
	perm.splice(src, 1);
	perm.splice(dst, 0, src);
	return transpose$1(t, perm);
}
function moveBatchAxis(axisSize, src, dst, x) {
	if (src === null) {
		const targetShape = [...x.shape];
		targetShape.splice(dst, 0, axisSize);
		return broadcast(x, targetShape, [dst]);
	} else if (src === dst) return x;
	else return moveaxis$1(x, src, dst);
}
var BatchTracer = class extends Tracer {
	val;
	batchDim;
	constructor(trace, val, batchDim) {
		super(trace);
		this.val = val;
		this.batchDim = batchDim;
	}
	get aval() {
		if (this.batchDim === null) return this.val.aval;
		else return mappedAval(this.batchDim, this.val.aval);
	}
	toString() {
		return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
	}
	get ref() {
		this.val.ref;
		return this;
	}
	dispose() {
		this.val.dispose();
	}
	fullLower() {
		if (this.batchDim === null) return this.val.fullLower();
		else return this;
	}
};
var BatchTrace = class extends Trace {
	pure(val) {
		return this.lift(pureArray(val));
	}
	lift(val) {
		return new BatchTracer(this, val, null);
	}
	processPrimitive(primitive, tracers, params) {
		const [valsIn, bdimsIn] = require_backend.unzip2(tracers.map((t) => [t.val, t.batchDim]));
		const vmapRule = vmapRules[primitive];
		if (vmapRule === void 0) throw new Error(`No vmap rule for: ${primitive}`);
		if (bdimsIn.every((d) => d === null)) return bind(primitive, valsIn, params).map((x) => new BatchTracer(this, x, null));
		const [valOuts, bdimOuts] = vmapRule(this.axisSize, valsIn, bdimsIn, params);
		if (valOuts.length !== bdimOuts.length) throw new Error(`vmap rule for ${primitive} returned mismatched lengths: ${valOuts.length} vs ${bdimOuts.length}`);
		return require_backend.zip(valOuts, bdimOuts).map(([x, bd]) => new BatchTracer(this, x, bd));
	}
	get axisSize() {
		return this.main.globalData;
	}
};
/**
* Process a primitive with built-in broadcasting.
*
* Reference: https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/interpreters/batching.py#L1029
*/
function broadcastBatcher(prim) {
	return (axisSize, args, dims, params) => {
		if (args.length === 0) throw new Error("Empty list in broadcastBatcher");
		const nd = Math.max(...args.map((x, i) => ndim$1(x) + (dims[i] === null ? 1 : 0)));
		const firstIdx = dims.findIndex((d) => d !== null);
		const firstBdim = dims[firstIdx] - args[firstIdx].ndim;
		if (require_backend.zip(args, dims).every(([x, d]) => d === null && ndim$1(x) < -firstBdim || d !== null && d - x.ndim === firstBdim)) return [[bind1(prim, args, params)], [nd + firstBdim]];
		args = args.map((x, i) => {
			if (dims[i] === null) return x;
			x = moveBatchAxis(axisSize, dims[i], 0, x);
			if (x.ndim < nd) x = x.reshape([
				x.shape[0],
				...require_backend.rep(nd - x.ndim, 1),
				...x.shape.slice(1)
			]);
			return x;
		});
		return [[bind1(prim, args, params)], [0]];
	};
}
function unopBatcher(prim) {
	return (axisSize, [x], [xBdim], params) => {
		return [[bind1(prim, [x], params)], [xBdim]];
	};
}
function lastDimsBatcher(prim, inputDims, numOutputs = 1) {
	return (axisSize, [x], [xBdim], params) => {
		if (xBdim < x.ndim - inputDims) return [bind(prim, [x], params), require_backend.rep(numOutputs, xBdim)];
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		return [bind(prim, [x], params), require_backend.rep(numOutputs, 0)];
	};
}
function batchIndexArrays(axisSize, indices, batchDims) {
	const rank = Math.max(...indices.map((index, i) => index.ndim + (batchDims[i] === null ? 1 : 0)));
	return [indices.map((index, i) => {
		const batchDim = batchDims[i];
		if (batchDim === null) return index;
		index = moveBatchAxis(axisSize, batchDim, 0, index);
		if (index.ndim < rank) index = index.reshape([
			index.shape[0],
			...require_backend.rep(rank - index.ndim, 1),
			...index.shape.slice(1)
		]);
		return index;
	}), rank];
}
const vmapRules = {
	["add"]: broadcastBatcher("add"),
	["mul"]: broadcastBatcher("mul"),
	["idiv"]: broadcastBatcher("idiv"),
	["mod"]: broadcastBatcher("mod"),
	["min"]: broadcastBatcher("min"),
	["max"]: broadcastBatcher("max"),
	["bit_combine"]: broadcastBatcher("bit_combine"),
	["bit_shift"]: broadcastBatcher("bit_shift"),
	["bit_count"]: unopBatcher("bit_count"),
	["neg"]: unopBatcher("neg"),
	["reciprocal"]: unopBatcher("reciprocal"),
	["floor"]: unopBatcher("floor"),
	["ceil"]: unopBatcher("ceil"),
	["stop_gradient"]: unopBatcher("stop_gradient"),
	["cast"]: unopBatcher("cast"),
	["bitcast"]: unopBatcher("bitcast"),
	["sin"]: unopBatcher("sin"),
	["cos"]: unopBatcher("cos"),
	["asin"]: unopBatcher("asin"),
	["atan"]: unopBatcher("atan"),
	["exp"]: unopBatcher("exp"),
	["log"]: unopBatcher("log"),
	["erf"]: unopBatcher("erf"),
	["erfc"]: unopBatcher("erfc"),
	["sqrt"]: unopBatcher("sqrt"),
	["reduce"](axisSize, [x], [xBdim], { op, axis }) {
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
		return [[reduce(x, op, newAxis)], [outBdim]];
	},
	["dot"](axisSize, [x, y], [xBdim, yBdim]) {
		x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
		y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
		const z = dot$2(x, y);
		return [[z], [z.ndim - 1]];
	},
	["conv"](axisSize, [x, y], [xBdim, yBdim], params) {
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		y = moveBatchAxis(axisSize, yBdim, 0, y);
		return [[conv$1(x, y, {
			...params,
			vmapDims: params.vmapDims + 1
		})], [0]];
	},
	["compare"]: broadcastBatcher("compare"),
	["where"]: broadcastBatcher("where"),
	["concatenate"](axisSize, xs, xBdims, { axis }) {
		const minBdim = Math.min(...xBdims.filter((d) => d !== null));
		xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
		const newAxis = axis + (minBdim <= axis ? 1 : 0);
		return [[concatenate$1(xs, newAxis)], [minBdim]];
	},
	["split"](axisSize, [x], [xBdim], { axis, sizes }) {
		const outs = split$2(x, axis + (xBdim <= axis ? 1 : 0), sizes);
		return [outs, require_backend.rep(outs.length, xBdim)];
	},
	["random_bits"](axisSize, [k0, k1], [bdim0, bdim1], { shape, mode }) {
		k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
		k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
		return [[randomBits(k0, k1, [axisSize, ...shape], mode)], [0]];
	},
	["gather"](axisSize, [x, ...indices], [xBdim, ...indicesBdim], { axis, outDim, uniqueIndices }) {
		if (indicesBdim.every((d) => d === null)) {
			const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
			let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
			let newOutDim = outDim;
			if (newOutDim < newBdim) newBdim += axis.length;
			else newOutDim += 1;
			return [[gather(x, indices, newAxis, newOutDim, uniqueIndices)], [newBdim]];
		}
		const [batchedIndices, indexRank] = batchIndexArrays(axisSize, indices, indicesBdim);
		indices = batchedIndices;
		if (xBdim === null) return [[gather(x, indices, axis, outDim, false)], [outDim]];
		else {
			x = moveBatchAxis(axisSize, xBdim, 0, x);
			const newAxis = [0, ...axis.map((ax) => ax + 1)];
			const extraBatchIndex = arange(axisSize).reshape([-1, ...require_backend.rep(indexRank - 1, 1)]);
			indices.splice(0, 0, extraBatchIndex);
			return [[gather(x, indices, newAxis, outDim, uniqueIndices)], [outDim]];
		}
	},
	["transpose"](axisSize, [x], [xBdim], { perm }) {
		const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
		newPerm.splice(xBdim, 0, xBdim);
		return [[transpose$1(x, newPerm)], [xBdim]];
	},
	["broadcast"](axisSize, [x], [xBdim], { shape, axis }) {
		return [[broadcast(x, shape.toSpliced(xBdim, 0, axisSize), axis.map((ax) => ax + (xBdim <= ax ? 1 : 0)))], [xBdim]];
	},
	["reshape"](axisSize, [x], [xBdim], { shape }) {
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		return [[reshape$1(x, [axisSize, ...shape])], [0]];
	},
	["flip"](axisSize, [x], [xBdim], { axis }) {
		return [[flip$1(x, axis.map((ax) => ax + (xBdim <= ax ? 1 : 0)))], [xBdim]];
	},
	["shrink"](axisSize, [x], [xBdim], { slice }) {
		return [[shrink(x, slice.toSpliced(xBdim, 0, [0, axisSize]))], [xBdim]];
	},
	["pad"](axisSize, [x], [xBdim], { width }) {
		return [[pad$1(x, width.toSpliced(xBdim, 0, [0, 0]))], [xBdim]];
	},
	["sort"]: lastDimsBatcher("sort", 1),
	["argsort"]: lastDimsBatcher("argsort", 1, 2),
	["scatter"]: (axisSize, [updates, ...indices], [updatesBdim, ...indicesBdim], { op, shape, axis, outDim, uniqueIndices }) => {
		const [batchedIndices, indexRank] = batchIndexArrays(axisSize, indices, indicesBdim);
		indices = batchedIndices;
		updates = moveBatchAxis(axisSize, updatesBdim, outDim, updates);
		const batchIndex = arange(axisSize).reshape([axisSize, ...require_backend.rep(indexRank - 1, 1)]);
		return [[scatter(updates, [batchIndex, ...indices], {
			op,
			shape: [axisSize, ...shape],
			axis: [0, ...axis.map((a) => a + 1)],
			outDim,
			uniqueIndices
		})], [0]];
	},
	["triangular_solve"](axisSize, [a, b], [aBdim, bBdim], { unitDiagonal }) {
		if (aBdim === null) {
			b = moveBatchAxis(axisSize, bBdim, -3, b);
			const [s, m, n] = b.shape.slice(-3);
			b = b.reshape([
				...b.shape.slice(0, -3),
				s * m,
				n
			]);
			let x = bind1("triangular_solve", [a, b], { unitDiagonal });
			x = x.reshape([
				...b.shape.slice(0, -2),
				s,
				m,
				n
			]);
			return [[x], [x.ndim - 3]];
		}
		a = moveBatchAxis(axisSize, aBdim, 0, a);
		b = moveBatchAxis(axisSize, bBdim, 0, b);
		return [[bind1("triangular_solve", [a, b], { unitDiagonal })], [0]];
	},
	["cholesky"]: lastDimsBatcher("cholesky", 2),
	["lu"]: lastDimsBatcher("lu", 2, 3),
	["jacobi_eigh"]: lastDimsBatcher("jacobi_eigh", 2, 2),
	["fft"](axisSize, [real, imag], [realBdim, imagBdim], params) {
		real = moveBatchAxis(axisSize, realBdim, 0, real);
		imag = moveBatchAxis(axisSize, imagBdim, 0, imag);
		return [bind("fft", [real, imag], params), [0, 0]];
	},
	["jit"](axisSize, args, dims, { name, jaxpr }) {
		const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
		const outs = bind("jit", [...newJaxpr.consts.map((c) => c.ref), ...args], {
			name: `${name}_vmap`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		return [outs, require_backend.rep(outs.length, 0)];
	}
};
const vmapJaxprCache = /* @__PURE__ */ new Map();
function vmapJaxpr(jaxpr, axisSize, dims) {
	const cacheKey = JSON.stringify([axisSize, dims]);
	const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const inAvals = jaxpr.inBinders.map((v, i) => {
		if (dims[i] === null) return v.aval;
		const shape = [...v.aval.shape];
		shape.splice(dims[i], 0, axisSize);
		return new ShapedArray(shape, v.aval.dtype, v.aval.weakType);
	});
	const { jaxpr: newJaxpr } = makeJaxpr$1((args) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(inAvals);
	if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	vmapJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vmapFlat(f, inAxes, args) {
	let axisSize = void 0;
	for (let i = 0; i < args.length; i++) if (inAxes[i] !== null) {
		const arg = args[i];
		if (!(arg instanceof Tracer)) throw new TypeError("vmap requires Tracer argument for mapped axes");
		const size = arg.shape[inAxes[i]];
		if (axisSize === void 0) axisSize = size;
		else if (axisSize !== size) throw new TypeError("vmap requires all mapped axes to have the same size");
	}
	if (axisSize === void 0) throw new TypeError("vmap requires at least one mapped axis");
	let valsOut, bdimsOut;
	try {
		var _usingCtx$2 = _usingCtx();
		const trace = new BatchTrace(_usingCtx$2.u(newMain(BatchTrace, axisSize)));
		const tracersOut = f(...args.map((x, i) => inAxes[i] === null ? pureArray(x) : new BatchTracer(trace, pureArray(x), inAxes[i]))).map((out) => fullRaise(trace, out));
		[valsOut, bdimsOut] = require_backend.unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
	} catch (_) {
		_usingCtx$2.e = _;
	} finally {
		_usingCtx$2.d();
	}
	return require_backend.zip(valsOut, bdimsOut).map(([valOut, bdim]) => moveBatchAxis(axisSize, bdim, 0, valOut));
}
function vmap$1(f, inAxes = 0) {
	return (...args) => {
		const [argsFlat, inTree] = flatten(args);
		let inAxesFlat = [];
		if (typeof inAxes === "number") inAxesFlat = require_backend.rep(argsFlat.length, inAxes);
		else for (let i = 0; i < args.length; i++) if (inAxes[i] == null) inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, null));
		else if (typeof inAxes[i] === "number") inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, inAxes[i]));
		else {
			const [axesFlat, axesTreeDef] = flatten(inAxes[i]);
			if (!inTree.childTreedefs[i].equals(axesTreeDef)) throw new TreeMismatchError("vmap", inTree.childTreedefs[i], axesTreeDef);
			inAxesFlat.push(...axesFlat);
		}
		const [fFlat, outTree] = flattenFun(f, inTree);
		const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
		if (outTree.value === void 0) throw new Error("outTree was not set in vmap");
		return unflatten(outTree.value, outsFlat);
	};
}
function jacfwd$1(f) {
	return function jacobianForward(x) {
		if (x.shape.length !== 1) throw new TypeError("jacfwd only supports 1D inputs");
		const [size] = x.shape;
		const pushfwd = (v) => jvp$1(f, [x], [v])[1];
		return vmap$1(pushfwd, [0])(eye(size, void 0, { dtype: x.dtype }));
	};
}
//#endregion
//#region src/frontend/jvp.ts
var JVPTracer = class extends Tracer {
	primal;
	tangent;
	constructor(trace, primal, tangent) {
		super(trace);
		this.primal = primal;
		this.tangent = tangent;
	}
	get aval() {
		return this.primal.aval;
	}
	toString() {
		return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
	}
	get ref() {
		this.primal.ref, this.tangent.ref;
		return this;
	}
	dispose() {
		this.primal.dispose();
		this.tangent.dispose();
	}
};
var JVPTrace = class extends Trace {
	pure(val) {
		return this.lift(pureArray(val));
	}
	lift(val) {
		return new JVPTracer(this, val, zerosLike$1(val.ref));
	}
	processPrimitive(primitive, tracers, params) {
		const [primalsIn, tangentsIn] = require_backend.unzip2(tracers.map((x) => [x.primal, x.tangent]));
		const jvpRule = jvpRules[primitive];
		if (jvpRule === void 0) throw new Error(`No JVP rule for: ${primitive}`);
		const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
		return require_backend.zip(primalsOut, tangentsOut).map(([x, t]) => new JVPTracer(this, x, t));
	}
};
/** Rule that applies the same operation to primals and tangents. */
function linearTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		return [bind(primitive, primals, params), bind(primitive, tangents, params)];
	};
}
/** Rule for product of gradients in bilinear operations. */
function bilinearTangentsJvp(primitive) {
	return ([x, y], [dx, dy], params) => {
		const primal = bind1(primitive, [x.ref, y.ref], params);
		const tangent = bind1(primitive, [x, dy], params).add(bind1(primitive, [dx, y], params));
		return [[primal], [tangent]];
	};
}
/** Rule that zeros out any tangents. */
function zeroTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		for (const t of tangents) t.dispose();
		const ys = bind(primitive, primals, params);
		return [ys, ys.map((y) => zerosLike$1(y.ref))];
	};
}
/** Gather values along the final axis while preserving leading coordinates. */
function takeAlongLastAxis(x, indices) {
	const coords = [];
	for (let axis = 0; axis < x.ndim - 1; axis++) {
		const shape = require_backend.rep(x.ndim, 1);
		shape[axis] = x.shape[axis];
		coords.push(arange(x.shape[axis]).reshape(shape));
	}
	coords.push(indices);
	return gather(x, coords, require_backend.range(x.ndim), 0, true);
}
/** Compute `a @ b.T`, batched to last two axes. */
function batchMatmulT(a, b) {
	return dot$2(a.reshape(a.shape.toSpliced(-1, 0, 1)), b.reshape(b.shape.toSpliced(-2, 0, 1)));
}
/** Batch matrix transpose. */
function mT(a) {
	return moveaxis$1(a, -2, -1);
}
function sliceAxis(a, axis, p) {
	const slices = Array(a.shape.length).fill([]);
	slices[require_backend.checkAxis(axis, a.ndim)] = p;
	return a.slice(...slices);
}
function padAxis(a, axis, p) {
	const pads = Array(a.shape.length).fill([0, 0]);
	pads[require_backend.checkAxis(axis, a.ndim)] = p;
	return pad$1(a, pads);
}
const jvpRules = {
	["add"]: linearTangentsJvp("add"),
	["mul"]: bilinearTangentsJvp("mul"),
	["idiv"]: zeroTangentsJvp("idiv"),
	["mod"]([x, y], [dx, dy]) {
		if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) {
			dx.dispose();
			dy.dispose();
			return [[x.ref, y.ref], [zerosLike$1(x), zerosLike$1(y)]];
		}
		const q = idiv(x.ref, y.ref);
		return [[mod(x, y)], [dx.sub(dy.mul(q))]];
	},
	["min"]([x, y], [dx, dy]) {
		return [[min$1(x.ref, y.ref)], [where$1(less$1(y, x), dy, dx)]];
	},
	["max"]([x, y], [dx, dy]) {
		return [[max$1(x.ref, y.ref)], [where$1(less$1(x, y), dy, dx)]];
	},
	["bit_combine"]: zeroTangentsJvp("bit_combine"),
	["bit_shift"]: zeroTangentsJvp("bit_shift"),
	["bit_count"]: zeroTangentsJvp("bit_count"),
	["neg"]: linearTangentsJvp("neg"),
	["reciprocal"]([x], [dx]) {
		const xRecip = reciprocal$1(x.ref);
		return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
	},
	["floor"]: zeroTangentsJvp("floor"),
	["ceil"]: zeroTangentsJvp("ceil"),
	["stop_gradient"]: zeroTangentsJvp("stop_gradient"),
	["cast"]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		if (require_backend.isFloatDtype(dtype) && require_backend.isFloatDtype(x.dtype)) return [[cast(x, dtype)], [cast(dx, dtype)]];
		else {
			dx.dispose();
			return [[cast(x.ref, dtype)], [zerosLike$1(x)]];
		}
	},
	["bitcast"]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		dx.dispose();
		return [[bitcast(x.ref, dtype)], [zerosLike$1(x)]];
	},
	["sin"]([x], [dx]) {
		return [[sin$1(x.ref)], [cos$1(x).mul(dx)]];
	},
	["cos"]([x], [dx]) {
		return [[cos$1(x.ref)], [neg(sin$1(x)).mul(dx)]];
	},
	["asin"]([x], [dx]) {
		const denom = sqrt$1(reciprocal$1(cast(1, x.dtype).sub(x.ref.mul(x.ref))));
		return [[asin$1(x)], [denom.mul(dx)]];
	},
	["atan"]([x], [dx]) {
		const denom = cast(1, x.dtype).add(x.ref.mul(x.ref));
		return [[atan$1(x)], [dx.div(denom)]];
	},
	["exp"]([x], [dx]) {
		const z = exp$1(x);
		return [[z.ref], [z.mul(dx)]];
	},
	["log"]([x], [dx]) {
		return [[log$1(x.ref)], [reciprocal$1(x).mul(dx)]];
	},
	["erf"]([x], [dx]) {
		const coeff = 2 / Math.sqrt(Math.PI);
		const expTerm = exp$1(neg(x.ref.mul(x.ref)));
		return [[erf$1(x)], [expTerm.mul(coeff).mul(dx)]];
	},
	["erfc"]([x], [dx]) {
		const coeff = -2 / Math.sqrt(Math.PI);
		const expTerm = exp$1(neg(x.ref.mul(x.ref)));
		return [[erfc$1(x)], [expTerm.mul(coeff).mul(dx)]];
	},
	["sqrt"]([x], [dx]) {
		const z = sqrt$1(x);
		return [[z.ref], [reciprocal$1(z.mul(2)).mul(dx)]];
	},
	["reduce"]([x], [dx], { op, axis }) {
		if (op === "Add") return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
		else if (op === "Mul") {
			const primal = reduce(x.ref, op, axis);
			const tangent = broadcast(primal.ref, x.shape, axis).mul(reciprocal$1(x)).mul(dx).sum(axis);
			return [[primal], [tangent]];
		} else if (op === "Min" || op === "Max") {
			const primal = reduce(x.ref, op, axis);
			const notMin = notEqual$1(x, broadcast(primal.ref, x.shape, axis));
			const minCount = where$1(notMin.ref, 0, 1).sum(axis);
			const tangent = where$1(notMin, 0, dx).sum(axis).div(minCount);
			return [[primal], [tangent]];
		} else throw new Error(`JVP rule not implemented for reduce op: ${op}`);
	},
	["pool"]: linearTangentsJvp("pool"),
	["pool_transpose"]: linearTangentsJvp("pool_transpose"),
	["dot"]: bilinearTangentsJvp("dot"),
	["conv"]: bilinearTangentsJvp("conv"),
	["compare"]: zeroTangentsJvp("compare"),
	["where"]([cond, x, y], [dcond, dx, dy]) {
		dcond.dispose();
		return [[where$1(cond.ref, x, y)], [where$1(cond, dx, dy)]];
	},
	["concatenate"]: linearTangentsJvp("concatenate"),
	["split"]: linearTangentsJvp("split"),
	["random_bits"]: zeroTangentsJvp("random_bits"),
	["gather"]([x, ...indices], [dx, ...dindices], params) {
		dindices.forEach((index) => index.dispose());
		const tangentIndices = indices.map((index) => index.ref);
		return [bind("gather", [x, ...indices], params), bind("gather", [dx, ...tangentIndices], params)];
	},
	["transpose"]: linearTangentsJvp("transpose"),
	["broadcast"]: linearTangentsJvp("broadcast"),
	["reshape"]: linearTangentsJvp("reshape"),
	["flip"]: linearTangentsJvp("flip"),
	["shrink"]: linearTangentsJvp("shrink"),
	["pad"]: linearTangentsJvp("pad"),
	["sort"]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		return [[y], [takeAlongLastAxis(dx, idx)]];
	},
	["argsort"]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		return [[y, idx.ref], [takeAlongLastAxis(dx, idx.ref), zerosLike$1(idx)]];
	},
	["scatter"]([updates, ...indices], [dupdates, ...dindices], params) {
		dindices.forEach((index) => index.dispose());
		const tangentIndices = indices.map((index) => index.ref);
		return [[scatter(updates, indices, params)], [scatter(dupdates, tangentIndices, params)]];
	},
	["triangular_solve"]([a, b], [da, db], { unitDiagonal }) {
		const x = triangularSolve$1(a.ref, b, { unitDiagonal });
		da = unitDiagonal ? triu(da, 1) : triu(da);
		const dax = batchMatmulT(da, x.ref);
		const dx = triangularSolve$1(a, db.sub(mT(dax)), { unitDiagonal });
		return [[x], [dx]];
	},
	["cholesky"]([a], [da]) {
		const L = cholesky$2(a.ref);
		da = da.ref.add(mT(da)).mul(.5);
		const W = triangularSolve$1(L.ref, da, { lower: true });
		const ST = triangularSolve$1(L.ref, mT(W), { lower: true });
		const dL = batchMatmulT(L.ref, triu(ST.ref, 1).add(triu(ST)).mul(.5));
		return [[L], [dL]];
	},
	["lu"]([a], [da]) {
		const [luMatrix, pivots, permutation] = lu$1(a);
		const [m, n] = a.shape.slice(-2);
		const k = Math.min(m, n);
		const lLower = tril(sliceAxis(luMatrix.ref, -1, [0, k]), -1);
		const L = (m > k ? padAxis(lLower, -1, [0, m - k]) : lLower).add(eye(m));
		const uUpper = triu(sliceAxis(luMatrix.ref, -2, [0, k]));
		const uPadded = n > k ? padAxis(uUpper, -2, [0, n - k]) : uUpper;
		const uEye = n > k ? padAxis(padAxis(eye(n - k), -1, [k, 0]), -2, [k, 0]) : zerosLike$1(uPadded.ref);
		const U = uPadded.add(uEye);
		const pda = batchMatmulT(permutation.ref.reshape([...permutation.shape, 1]).equal(arange(m)).astype(da.dtype), mT(da));
		const la = mT(triangularSolve$1(L.ref, mT(pda), {
			lower: true,
			unitDiagonal: true
		}));
		const lau = triangularSolve$1(mT(U.ref), la, { lower: true });
		const lDot = batchMatmulT(L, mT(tril(lau.ref, -1)));
		const uDot = batchMatmulT(triu(lau), mT(U));
		return [[
			luMatrix,
			pivots,
			permutation
		], [
			lDot.add(uDot),
			zerosLike$1(pivots.ref),
			zerosLike$1(permutation.ref)
		]];
	},
	["jacobi_eigh"]() {
		throw new Error("JVP rule not implemented for jacobi_eigh");
	},
	["fft"]: linearTangentsJvp("fft"),
	["jit"](primals, tangents, { name, jaxpr }) {
		const newJaxpr = jvpJaxpr(jaxpr);
		const outs = bind("jit", [
			...newJaxpr.consts.map((c) => c.ref),
			...primals,
			...tangents
		], {
			name: `${name}_jvp`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		const n = outs.length / 2;
		if (!Number.isInteger(n)) throw new Error("internal: JVP Jaxpr output length is not even");
		const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
		return [primalsOut, tangentsOut];
	}
};
const jvpJaxprCache = /* @__PURE__ */ new Map();
function jvpJaxpr(jaxpr) {
	if (jvpJaxprCache.has(jaxpr)) return jvpJaxprCache.get(jaxpr);
	const inAvals = jaxpr.inBinders.map((v) => v.aval);
	const { jaxpr: newJaxpr } = makeJaxpr$1((primals, tangents) => jvpFlat(jaxprAsFun(jaxpr), primals, tangents))(inAvals, inAvals);
	jvpJaxprCache.set(jaxpr, newJaxpr);
	return newJaxpr;
}
function jvpFlat(f, primals, tangents) {
	try {
		var _usingCtx$1 = _usingCtx();
		const trace = new JVPTrace(_usingCtx$1.u(newMain(JVPTrace)));
		return require_backend.unzip2(f(...require_backend.zip(primals, tangents).map(([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t)))).map((out) => fullRaise(trace, out)).map((t) => [t.primal, t.tangent]));
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
function jvp$1(f, primals, tangents, { hasAux = false } = {}) {
	const [primalsFlat, inTree] = flatten(primals);
	const [tangentsFlat, inTree2] = flatten(tangents);
	if (!inTree.equals(inTree2)) throw new TreeMismatchError("jvp", inTree, inTree2);
	let flatFun, outTree, aux;
	if (hasAux) [flatFun, outTree, aux] = flattenFunWithAux(f, inTree);
	else [flatFun, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, tangentsOutFlat] = jvpFlat(flatFun, primalsFlat, tangentsFlat);
	if (outTree.value === void 0) throw new Error("outTree was not set in jvp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const tangentsOut = unflatten(outTree.value, tangentsOutFlat);
	if (hasAux) return [
		primalsOut,
		tangentsOut,
		lowerAux(aux.value)
	];
	return [primalsOut, tangentsOut];
}
/** Lowering for auxiliary data returned in `hasAux: true` methods. */
function lowerAux(aux) {
	const level = currentTraceLevel();
	return map((x) => {
		if (x instanceof Tracer) while (x._trace.main.level > level) if (x instanceof JVPTracer) {
			x.tangent.dispose();
			x = x.primal;
		} else {
			const y = x.fullLower();
			if (y._trace.main.level >= x._trace.main.level) throw new Error("internal: lowerAux did not reduce trace level");
			x = y;
		}
		return x;
	}, aux);
}
//#endregion
//#region src/frontend/linearize.ts
/** @file Implementations of vjp() and partial evaluation. */
/** Array value that can either be known or unknown. */
var PartialVal = class PartialVal {
	val;
	aval;
	constructor(val, aval) {
		this.val = val;
		this.aval = aval;
	}
	static known(val) {
		return new PartialVal(val, ShapedArray.fromAval(val.aval));
	}
	static unknown(aval) {
		return new PartialVal(null, ShapedArray.fromAval(aval));
	}
	get isKnown() {
		return this.val !== null;
	}
	toString() {
		return this.val ? this.val.toString() : this.aval.toString();
	}
};
function partialEvalFlat(f, pvalsIn) {
	const trace = new PartialEvalTrace(newMain(PartialEvalTrace));
	const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
	const unknownTracersIn = tracersIn.filter((t) => !t.pval.isKnown).map((t) => t.ref);
	const tracersOut = f(...tracersIn).map((out) => fullRaise(trace, out));
	const pvalsOut = tracersOut.map((t) => t.pval);
	return {
		jaxpr: partialEvalGraphToJaxpr(unknownTracersIn, tracersOut.filter((t) => !t.pval.isKnown)),
		pvalsOut
	};
}
/**
* Helper function with shared Jaxpr logic between linearize and vjp.
*
* Internally, vjp() looks very similar to linearize() but returns a function
* evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
* of pushing forward tangents.
*/
function linearizeFlatUtil(f, primalsIn) {
	const pvalsIn = [...primalsIn.map(PartialVal.known), ...primalsIn.map((t) => PartialVal.unknown(t.aval))];
	const fJvp = (...x) => {
		const k = x.length / 2;
		const [primalsOut, tangentsOut] = jvp$1(f, x.slice(0, k), x.slice(k, 2 * k));
		return [...primalsOut, ...tangentsOut];
	};
	const { jaxpr, pvalsOut } = partialEvalFlat(fJvp, pvalsIn);
	const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
	if (!primalPvals.every((pval) => pval.isKnown)) throw new Error("Not all primal values are known after partial evaluation");
	return {
		primalsOut: primalPvals.map((pval) => pval.val),
		jaxpr
	};
}
function linearizeFlat(f, primalsIn) {
	const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
	const fLin = (...tangents) => evalJaxpr(jaxpr.jaxpr, [...jaxpr.consts.map((c) => c.ref), ...tangents]);
	const dispose = () => jaxpr.dispose();
	return [
		primalsOut,
		fLin,
		dispose
	];
}
function linearize$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, fLinFlat, dispose] = linearizeFlat(fFlat, primalsInFlat.map(pureArray));
	if (outTree.value === void 0) throw new Error("outTree was not set in linearize");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fLin = ((...tangentsIn) => {
		const [tangentsInFlat, inTree2] = flatten(tangentsIn);
		if (!inTree.equals(inTree2)) throw new TreeMismatchError("linearize", inTree, inTree2);
		const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
		return unflatten(outTree.value, tangentsOutFlat);
	});
	fLin.dispose = dispose;
	if (hasAux) return [
		primalsOut,
		fLin,
		lowerAux(aux.value)
	];
	return [primalsOut, fLin];
}
var PartialEvalTracer = class extends Tracer {
	pval;
	recipe;
	#rc;
	constructor(trace, pval, recipe) {
		super(trace);
		this.pval = pval;
		this.recipe = recipe;
		this.#rc = 1;
	}
	get aval() {
		return this.pval.aval;
	}
	toString() {
		if (!this.recipe) return `PartialEvalTracer(${this.pval.toString()})`;
		else return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
	}
	get ref() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc++;
		return this;
	}
	dispose() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		if (--this.#rc === 0) {
			if (this.pval.isKnown) this.pval.val.dispose();
			else if (this.recipe) {
				if (this.recipe.type === "Const") this.recipe.val.dispose();
				else if (this.recipe.type === "JaxprEqn") this.recipe.tracersIn.forEach((t) => t.dispose());
			}
		}
	}
	fullLower() {
		if (this.pval.isKnown) {
			const val = this.pval.val.ref;
			this.dispose();
			return val;
		}
		return this;
	}
};
var PartialEvalTrace = class extends Trace {
	newArg(pval) {
		if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
		return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
	}
	pure(val) {
		return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
	}
	lift = this.pure;
	instantiateConst(tracer) {
		if (!tracer.pval.isKnown) return tracer;
		else {
			const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
			const val = tracer.pval.val.ref;
			tracer.dispose();
			return new PartialEvalTracer(this, pval, {
				type: "Const",
				val
			});
		}
	}
	processPrimitive(primitive, tracers, params) {
		if (tracers.every((t) => t.pval.isKnown)) return bind(primitive, tracers.map((t) => t.fullLower()), params);
		if (primitive === "jit") {
			const { name, jaxpr, numConsts } = params;
			return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
		}
		const tracersIn = tracers.map((t) => this.instantiateConst(t));
		const avalsIn = tracersIn.map((t) => t.pval.aval);
		const avalsOut = abstractEvalRules[primitive](avalsIn, params);
		const recipe = {
			type: "JaxprEqn",
			prim: primitive,
			tracersIn,
			params,
			avalsOut,
			tracerRefsOut: []
		};
		const tracersOut = avalsOut.map((aval, i) => {
			if (i > 0) tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
		});
		recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
		return tracersOut;
	}
	/**
	* Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
	* values as possible (with JIT) and forwarding the unknown ones.
	*
	* Used when encountering a Jit rule during the trace.
	*/
	#partialEvalJaxpr(name, jaxpr, numConsts, tracers) {
		jaxpr = jaxpr.flatten();
		const inUnknowns = tracers.map((t) => !t.pval.isKnown);
		const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(jaxpr, inUnknowns);
		const [knownTracers, unknownTracers] = require_backend.partitionList(inUnknowns, tracers);
		const outs1Res = bind("jit", knownTracers.map((t) => t.fullLower()), {
			name: `${name}_peval`,
			jaxpr: jaxpr1,
			numConsts: 0
		});
		const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
		const recipe = {
			type: "JaxprEqn",
			prim: "jit",
			tracersIn: outs1Res.slice(jaxpr1.outs.length - numRes).map((x) => this.instantiateConst(fullRaise(this, x))).concat(unknownTracers),
			params: {
				name: `${name}_resid`,
				jaxpr: jaxpr2,
				numConsts: 0
			},
			avalsOut: jaxpr2.outs.map((x) => x.aval),
			tracerRefsOut: []
		};
		const outs2 = jaxpr2.outs.map((x, i) => {
			if (i > 0) recipe.tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
		});
		recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));
		let i = 0;
		let j = 0;
		return outUnknowns.map((unk) => unk ? outs2[j++] : outs1[i++]);
	}
};
/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(jaxpr, inUnknowns, instantiate) {
	jaxpr = jaxpr.flatten();
	const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
	const knownVars = new Set(knownIns);
	const residuals = /* @__PURE__ */ new Set();
	const eqns1 = [];
	const eqns2 = [];
	for (const eqn of jaxpr.eqns) {
		if (eqn.primitive === "jit") throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
		if (eqn.inputs.some((x) => x instanceof Var && !knownVars.has(x))) {
			for (const x of eqn.inputs) if (x instanceof Var && knownVars.has(x)) residuals.add(x);
			eqns2.push(eqn);
		} else {
			eqns1.push(eqn);
			for (const v of eqn.outBinders) knownVars.add(v);
		}
	}
	const outUnknowns = jaxpr.outs.map((x) => x instanceof Var && !knownVars.has(x));
	if (instantiate !== void 0) for (let i = 0; i < jaxpr.outs.length; i++) {
		const x = jaxpr.outs[i];
		if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
			residuals.add(x);
			outUnknowns[i] = true;
		}
	}
	const residualsL = Array.from(residuals);
	const [ins1, ins2] = require_backend.partitionList(inUnknowns, jaxpr.inBinders);
	const [outs1, outs2] = require_backend.partitionList(outUnknowns, jaxpr.outs);
	return {
		jaxpr1: new Jaxpr(ins1, eqns1, outs1.concat(residualsL)),
		jaxpr2: new Jaxpr(residualsL.concat(ins2), eqns2, outs2),
		outUnknowns,
		numRes: residualsL.length
	};
}
/**
* Convert the graph representation of a partial eval to a standard Jaxpr.
* Also called `tracers_to_jaxpr()` in JAX.
*/
function partialEvalGraphToJaxpr(tracersIn, tracersOut) {
	const tracerToVar = /* @__PURE__ */ new Map();
	const constToVar = /* @__PURE__ */ new Map();
	const processedEqns = /* @__PURE__ */ new Set();
	const eqns = [];
	for (const t of tracersIn) tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
	for (const t of require_backend.toposort(tracersOut, (t) => t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [])) {
		if (!t.recipe) throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
		if (t.recipe.type === "LambdaBinding") {
			if (!tracersIn.includes(t)) throw new TypeError("LambdaBinding tracer not in input list");
		} else if (t.recipe.type === "Const") {
			const val = t.recipe.val;
			let binder = constToVar.get(val);
			if (!binder) {
				binder = new Var(ShapedArray.fromAval(val.aval));
				constToVar.set(val, binder);
			}
			tracerToVar.set(t, binder);
		} else if (t.recipe.type === "JaxprEqn") {
			if (!processedEqns.has(t.recipe)) {
				processedEqns.add(t.recipe);
				const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t));
				const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
				for (let i = 0; i < outBinders.length; i++) {
					const tracerOut = t.recipe.tracerRefsOut[i].deref();
					if (tracerOut) tracerToVar.set(tracerOut, outBinders[i]);
				}
				eqns.push(new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders));
			}
		}
	}
	const [consts, constvars] = require_backend.unzip2(constToVar.entries());
	let jaxpr = new Jaxpr([...constvars, ...tracersIn.map((t) => tracerToVar.get(t))], eqns, tracersOut.map((t) => tracerToVar.get(t)));
	typecheckJaxpr(jaxpr);
	for (const t of consts) t.ref;
	for (const t of tracersIn) t.dispose();
	for (const t of tracersOut) t.dispose();
	jaxpr = jaxpr.simplify();
	if (require_backend.DEBUG >= 5) console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
	return new ClosedJaxpr(jaxpr, consts);
}
/** Marker type for pullback, used by transpose rules. */
var UndefPrimal = class {
	aval;
	constructor(aval) {
		this.aval = ShapedArray.fromAval(aval);
	}
};
/**
* Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
*
* Will raise a TypeError if the provided Jaxpr is not a linear function of its,
* inputs, as general expressions cannot be transposed.
*/
function evalJaxprTransposed(jaxpr, args, cotangents) {
	const knownPrimals = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (!(args[i] instanceof UndefPrimal)) knownPrimals.set(jaxpr.inBinders[i], args[i]);
	const ctStore = /* @__PURE__ */ new Map();
	const readCotangent = (v) => {
		const ct = ctStore.get(v);
		if (ct) {
			ctStore.delete(v);
			return ct;
		} else return zeros(v.aval.shape, { dtype: v.aval.dtype });
	};
	const writeCotangent = (v, ct) => {
		if (ct !== null) {
			const oldCt = ctStore.get(v);
			if (oldCt) ctStore.set(v, add$1(oldCt, ct));
			else ctStore.set(v, ct);
		}
	};
	for (let i = 0; i < jaxpr.outs.length; i++) {
		const v = jaxpr.outs[i];
		if (v instanceof Var) writeCotangent(v, cotangents[i]);
	}
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		const primalsIn = eqn.inputs.map((v) => v instanceof Lit ? array(v.value, { dtype: v.dtype }) : knownPrimals.has(v) ? knownPrimals.get(v).ref : new UndefPrimal(v.aval));
		const cotangentsOut = eqn.outBinders.map(readCotangent);
		const rule = transposeRules[eqn.primitive];
		if (!rule) throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
		const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
		for (let j = 0; j < eqn.inputs.length; j++) {
			const v = eqn.inputs[j];
			if (v instanceof Var && !knownPrimals.has(v)) writeCotangent(v, cotangentsIn[j]);
			else if (cotangentsIn[j] !== null) throw new Error("internal: cotangent should be null");
		}
	}
	for (const t of knownPrimals.values()) t.dispose();
	const results = [];
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (args[i] instanceof UndefPrimal) results.push(readCotangent(jaxpr.inBinders[i]));
	return results;
}
/**
* Inverse operation of `generalBroadcast()` for backpropagation.
*
* `x` has the shape of the result of an operation that was broadcasted with
* `target` (it's a cotangent during backprop). Returns a tracer with rank and
* shape equal to `target`.
*/
function unbroadcast(x, target) {
	const shape = target.aval.shape;
	const extraDims = x.ndim > shape.length ? require_backend.range(x.ndim - shape.length) : [];
	if (x.ndim < shape.length) throw new Error(`unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape})`);
	const unsqueeze = [];
	const keptReduceDims = [];
	for (let i = 0; i < shape.length; i++) {
		const indexFromEnd = shape.length - i;
		const indexInX = x.ndim - indexFromEnd;
		const xLen = x.shape[indexInX];
		if (xLen > 1 && shape[i] === 1) {
			unsqueeze.push(i);
			keptReduceDims.push(indexInX);
		} else if (shape[i] !== xLen) throw new Error("internal: unbroadcast shape mismatch");
	}
	const reductionDims = [...extraDims, ...keptReduceDims];
	if (reductionDims.length === 0) return x;
	let result = x.sum(reductionDims);
	if (!require_backend.deepEqual(result.shape, shape)) result = broadcast(result, shape, unsqueeze);
	return result;
}
/** Transpose a dot, treating every axis as either batch or contracting. */
function transposeDot(ct, known, target) {
	const broadcastShape = require_backend.generalBroadcast(target.aval.shape, known.shape);
	const ndim = broadcastShape.length;
	const align = (shape) => Array(ndim - shape.length).fill(1).concat(shape);
	const targetShape = align(target.aval.shape);
	const knownShape = align(known.shape);
	const contractingDims = require_backend.range(ndim).filter((axis) => targetShape[axis] === 1);
	const batchDims = require_backend.range(ndim).filter((axis) => targetShape[axis] !== 1);
	const perm = [...batchDims, ...contractingDims];
	const dotShape = (shape) => [...batchDims.map((axis) => shape[axis]), require_backend.prod(contractingDims.map((axis) => shape[axis]))];
	return dot$2(transpose$1(broadcast(ct, broadcastShape, [-1]), perm).reshape(dotShape(broadcastShape)), transpose$1(known.reshape(knownShape), perm).reshape(dotShape(knownShape))).reshape(target.aval.shape);
}
var NonlinearError = class extends TypeError {
	constructor(primitive) {
		super(`Nonlinear operation in backward pass for ${primitive}`);
	}
};
const transposeRules = {
	["mul"]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError("mul");
		return [x instanceof UndefPrimal ? unbroadcast(mul(ct, y), x) : null, y instanceof UndefPrimal ? unbroadcast(mul(x, ct), y) : null];
	},
	["neg"]([ct], [x]) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("neg");
		return [neg(ct)];
	},
	["add"]([ct], [x, y]) {
		if (!(x instanceof UndefPrimal || y instanceof UndefPrimal)) throw new NonlinearError("add");
		if (x instanceof UndefPrimal && y instanceof UndefPrimal) return [unbroadcast(ct.ref, x), unbroadcast(ct, y)];
		return x instanceof UndefPrimal ? (y.dispose(), [unbroadcast(ct, x), null]) : (x.dispose(), [null, unbroadcast(ct, y)]);
	},
	["reduce"]([ct], [x], { op, axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("reduce");
		if (op === "Add") return [broadcast(ct, x.aval.shape, axis)];
		else throw new NonlinearError("reduce");
	},
	["pool"]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("pool");
		return bind("pool_transpose", [ct], {
			inShape: x.aval.shape,
			window,
			strides
		});
	},
	["pool_transpose"]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("pool_transpose");
		return bind("pool", [ct], {
			window,
			strides
		});
	},
	["dot"]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError("dot");
		return x instanceof UndefPrimal ? [transposeDot(ct, y, x), null] : [null, transposeDot(ct, x, y)];
	},
	["conv"]([ct], [lhs, rhs], params) {
		if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal) throw new NonlinearError("conv");
		const v = params.vmapDims;
		const rev01 = [
			...require_backend.range(v),
			v + 1,
			v,
			...require_backend.range(v + 2, ct.ndim)
		];
		if (lhs instanceof UndefPrimal) {
			let kernel = rhs;
			kernel = transpose$1(kernel, rev01);
			kernel = flip$1(kernel, require_backend.range(v + 2, kernel.ndim));
			return [conv$1(ct, kernel, {
				vmapDims: v,
				strides: params.lhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedKernel = (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
					const padBefore = dilatedKernel - 1 - pl;
					return [padBefore, (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1 + dilatedKernel - 1 - dilatedCt - padBefore];
				}),
				lhsDilation: params.strides,
				rhsDilation: params.rhsDilation
			}), null];
		} else {
			let result = conv$1(transpose$1(lhs, rev01), transpose$1(ct, rev01), {
				vmapDims: v,
				strides: params.rhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedLhs = (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
					const dilatedKernel = (rhs.aval.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					return [pl, (ct.shape[i + v + 2] - 1) * params.strides[i] + 1 - dilatedLhs + (dilatedKernel - pl - 1)];
				}),
				lhsDilation: params.lhsDilation,
				rhsDilation: params.strides
			});
			result = transpose$1(result, rev01);
			return [null, result];
		}
	},
	["where"]([ct], [cond, x, y]) {
		const cts = [
			null,
			null,
			null
		];
		if (cond instanceof UndefPrimal) throw new NonlinearError("where");
		if (x instanceof UndefPrimal) cts[1] = unbroadcast(where$1(cond.ref, ct.ref, 0), x);
		else x.dispose();
		if (y instanceof UndefPrimal) cts[2] = unbroadcast(where$1(cond.ref, 0, ct.ref), y);
		else y.dispose();
		ct.dispose();
		cond.dispose();
		return cts;
	},
	["concatenate"]([ct], inputs, { axis }) {
		const cts = split$2(ct, axis, inputs.map((x) => x.aval.shape[axis]));
		return inputs.map((x, i) => {
			if (x instanceof UndefPrimal) return cts[i];
			x.dispose(), cts[i].dispose();
			return null;
		});
	},
	["split"](cts, [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("split");
		return [concatenate$1(cts, axis)];
	},
	["gather"]([ct], [x, ...indices], { axis, outDim, uniqueIndices }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("gather");
		if (indices.some((i) => i instanceof UndefPrimal)) throw new NonlinearError("gather");
		return [scatter(ct, indices, {
			op: "add",
			shape: x.aval.shape,
			axis,
			outDim,
			uniqueIndices
		}), ...indices.map(() => null)];
	},
	["transpose"]([ct], [x], { perm }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("transpose");
		return [transpose$1(ct, require_backend.invertPermutation(perm))];
	},
	["broadcast"]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("broadcast");
		let out = reduce(ct, "Add", axis);
		const expanded = x.aval.shape.flatMap((d, i) => d === 1 && out.shape[i] !== 1 ? [i] : []);
		if (expanded.length > 0) out = reduce(out, "Add", expanded, { keepdims: true });
		return [out];
	},
	["reshape"]([ct], [x], _) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("reshape");
		return [reshape$1(ct, x.aval.shape)];
	},
	["flip"]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("flip");
		return [flip$1(ct, axis)];
	},
	["shrink"]([ct], [x], { slice }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("shrink");
		return [pad$1(ct, slice.map(([s, e], i) => [s, x.aval.shape[i] - e]))];
	},
	["pad"]([ct], [x], { width }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError("pad");
		return [shrink(ct, width.map(([s, _e], i) => [s, s + x.aval.shape[i]]))];
	},
	["scatter"]([ct], [updates, ...indices], { axis, outDim, uniqueIndices }) {
		if (!(updates instanceof UndefPrimal)) throw new NonlinearError("scatter");
		if (indices.some((i) => i instanceof UndefPrimal)) throw new NonlinearError("scatter");
		return [gather(ct, indices, axis, outDim, uniqueIndices), ...indices.map(() => null)];
	},
	["triangular_solve"]([ct], [a, b], { unitDiagonal }) {
		if (a instanceof UndefPrimal || !(b instanceof UndefPrimal)) throw new NonlinearError("triangular_solve");
		return [null, triangularSolve$1(moveaxis$1(a, -2, -1), ct, {
			lower: true,
			unitDiagonal
		})];
	},
	["fft"]([ctReal, ctImag], [real, imag], params) {
		const aval = ShapedArray.fromAval(real.aval);
		const n = aval.shape[aval.ndim - 1];
		const [realAdjoint, imagAdjoint] = bind("fft", [ctReal, ctImag], {
			factors: params.factors,
			inverse: !params.inverse
		});
		const scale = params.inverse ? 1 / n : n;
		const realCt = mul(realAdjoint, scale);
		const imagCt = mul(imagAdjoint, scale);
		const cts = [null, null];
		if (real instanceof UndefPrimal) cts[0] = realCt;
		else realCt.dispose();
		if (imag instanceof UndefPrimal) cts[1] = imagCt;
		else imagCt.dispose();
		return cts;
	},
	["jit"](cts, args, { name, jaxpr }) {
		const undefPrimals = args.map((x) => x instanceof UndefPrimal);
		const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
		const residuals = args.filter((x, i) => !undefPrimals[i]);
		const outs = bind("jit", [
			...newJaxpr.consts.map((c) => c.ref),
			...residuals,
			...cts
		], {
			name: `${name}_t`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		let i = 0;
		return undefPrimals.map((isUndef) => isUndef ? outs[i++] : null);
	}
};
const transposeJaxprCache = /* @__PURE__ */ new Map();
function transposeJaxpr(jaxpr, undefPrimals) {
	const cacheKey = JSON.stringify(undefPrimals);
	const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
	const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
	const { jaxpr: newJaxpr } = makeJaxpr$1((forwardIn, cotangents) => {
		const args = [];
		let forwardInIdx = 0;
		for (let i = 0; i < undefPrimals.length; i++) if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
		else args.push(forwardIn[forwardInIdx++]);
		return evalJaxprTransposed(jaxpr, args, cotangents);
	})(forwardInTypes, outTypes);
	typecheckJaxpr(newJaxpr.jaxpr);
	if (!transposeJaxprCache.has(jaxpr)) transposeJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	transposeJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vjpFlat(f, primalsIn) {
	const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
	const fVjp = (...cotangents) => {
		const transposeInputs = [...jaxpr.consts.map((c) => c.ref), ...primalsIn.map((t) => new UndefPrimal(t.aval))];
		return evalJaxprTransposed(jaxpr.jaxpr, transposeInputs, cotangents);
	};
	const dispose = () => jaxpr.dispose();
	return [
		primalsOut,
		fVjp,
		dispose
	];
}
function vjp$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, fVjpFlat, dispose] = vjpFlat(fFlat, primalsInFlat.map(pureArray));
	if (outTree.value === void 0) throw new Error("outTree was not set in vjp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fVjp = ((cotangentsOut) => {
		const [cotangentsOutFlat, outTree2] = flatten(cotangentsOut);
		if (!outTree.value.equals(outTree2)) throw new TreeMismatchError("vjp", outTree.value, outTree2);
		return unflatten(inTree, fVjpFlat(...cotangentsOutFlat.map(pureArray)));
	});
	fVjp.dispose = dispose;
	if (hasAux) return [
		primalsOut,
		fVjp,
		lowerAux(aux.value)
	];
	return [primalsOut, fVjp];
}
function grad$1(f, opts) {
	const valueAndGradFn = valueAndGrad$1(f, opts);
	return (...x) => {
		if (opts?.hasAux) {
			const [[y, aux], dx] = valueAndGradFn(...x);
			y.dispose();
			return [dx, aux];
		} else {
			const [y, dx] = valueAndGradFn(...x);
			y.dispose();
			return dx;
		}
	};
}
function valueAndGrad$1(f, opts) {
	const argnums = opts?.argnums ?? 0;
	const hasAux = opts?.hasAux ?? false;
	require_backend.checkInts(argnums);
	const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
	return (...x) => {
		if (x.length === 0) throw new Error("grad requires at least one argument to differentiate");
		for (let i = 0; i < x.length; i++) if (!argnumsSet.has(i)) x[i] = map(stopGradient$1, x[i]);
		const [y, fVjp, aux] = vjp$1(f, x, { hasAux });
		if (!(y instanceof Tracer) || ndim$1(y) !== 0) throw new TypeError("grad requires a scalar output");
		if (!require_backend.isFloatDtype(y.dtype)) throw new TypeError("grad only supports floating-point dtypes");
		const cts = fVjp(onesLike$1(y.ref));
		fVjp.dispose();
		for (let i = 0; i < cts.length; i++) if (!argnumsSet.has(i)) dispose(cts[i]);
		const grads = typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
		return hasAux ? [[y, aux], grads] : [y, grads];
	};
}
function jacrev$1(f) {
	return function jacobianReverse(x) {
		if (x.shape.length !== 1) throw new TypeError("jacrev only supports 1D inputs");
		const [size] = x.shape;
		const pullback = (ct) => {
			const [y, fVjp] = vjp$1(f, [x]);
			y.dispose();
			const [ret] = fVjp(ct);
			fVjp.dispose();
			return ret;
		};
		return vmap$1(pullback, [1])(eye(size, void 0, { dtype: x.dtype }));
	};
}
function hessian$1(f) {
	return jacfwd$1(grad$1(f));
}
//#endregion
//#region src/library/numpy/einsum.ts
const bprod = (...xs) => xs.reduce((acc, x) => acc * BigInt(x), 1n);
const uniq = (arr) => Array.from(new Set(arr));
const EINSUM_COMPONENT_RE = /\p{ID_Start}|\.\.\./gu;
const einsumParseCache = /* @__PURE__ */ new Map();
/** Parse an Einstein notation string into a runnable `EinsumInput`. */
function parseEinsumExpression(expr, shapes) {
	return require_backend.runWithCache(einsumParseCache, [expr, shapes], () => {
		const idents = [...expr.split("->")[0].matchAll(EINSUM_COMPONENT_RE).map((m) => m[0]).filter((c) => c !== "...")];
		if (!expr.includes("->")) {
			const counts = /* @__PURE__ */ new Map();
			for (const c of idents) counts.set(c, (counts.get(c) ?? 0) + 1);
			const outputIndices = Array.from(counts.entries()).filter(([, count]) => count === 1).map(([char]) => char).sort();
			if (expr.includes("...")) outputIndices.splice(0, 0, "...");
			expr += "->" + outputIndices.join("");
		}
		const identToIndex = new Map(uniq(idents).sort().map((c, i) => [c, i]));
		const componentsToIndices = (components, rank) => components.flatMap((c) => {
			if (c === "...") {
				const start = rank !== void 0 ? components.length - 1 + ellipsisRank - rank : 0;
				return require_backend.range(identToIndex.size + start, identToIndex.size + ellipsisRank);
			}
			return identToIndex.get(c);
		});
		let ellipsisRank = 0;
		const [lhs, rhs] = expr.split("->");
		const lhsComponents = lhs.split(",").map((part) => [...part.matchAll(EINSUM_COMPONENT_RE).map((m) => m[0])]);
		const rhsComponents = [...rhs.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]);
		for (const [i, components] of lhsComponents.entries()) {
			const shape = shapes[i];
			const ellipsisIndex = components.indexOf("...");
			if (ellipsisIndex !== -1) {
				if (components.lastIndexOf("...") !== ellipsisIndex) throw new Error("Multiple ellipses in one einsum operand is not allowed");
				const numExplicit = components.length - 1;
				if (shape.length < numExplicit) throw new Error(`Einsum operand ${i} has shape ${JSON.stringify(shape)} but indexed with "${components.join("")}"`);
				ellipsisRank = Math.max(ellipsisRank, shape.length - numExplicit);
			}
		}
		return {
			shapes,
			lhsIndices: lhsComponents.map((components, i) => componentsToIndices(components, shapes[i].length)),
			rhsIndex: componentsToIndices(rhsComponents)
		};
	});
}
var EinsumPath = class {
	/** Parsed and normalized input for the einsum. */
	input;
	/** Mapping of each index number to its size in the shape array. */
	sizeMap;
	/**
	* A list of tensor contractions.
	*
	* This is ordered by operation order. Each entry corresponds to a single
	* elementwise product and/or inner contraction between two tensors, and it
	* contains the indices of the tensors to be contracted.
	*
	* The indices of input tensors are [0..n), and each intermediate from the
	* path at index i produces a new tensor at index n + i at the end
	* (opt_einsum internally calls this "SSA form").
	*
	* Invariants:
	* - Each group in the path consists of two tensors.
	* - For n input tensors, there are n-1 groups in the path.
	* - Every tensor must be in the path exactly once, except the final output.
	*
	* @example
	* Given einsum for `(A, B, C)`, this path corresponds to `(A, B)` and then
	* `(AB, C)`.
	* ```
	* [[0, 1], [3, 2]]
	* ```
	*/
	path;
	constructor(input, sizeMap, path) {
		this.input = input;
		this.sizeMap = sizeMap;
		this.path = path;
	}
	/** Shape of the final output tensor. */
	get outputShape() {
		return this.input.rhsIndex.map((i) => this.sizeMap.get(i));
	}
	/** Estimate the number of FLOPs to execute this einsum path. */
	get approximateFlops() {
		return approximatePathFlops(this.input, this.sizeMap, this.path);
	}
};
function approximatePathFlops(input, sizeMap, path) {
	if (path.length == 0) {
		const [indices] = input.lhsIndices;
		return bprod(...uniq(indices).map((i) => sizeMap.get(i)));
	}
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	let totalFlops = 0n;
	for (const tensorGroup of path) {
		const indexReduced = [];
		const indexGroup = [];
		for (const tensorIdx of tensorGroup) for (const idx of indices[tensorIdx]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		totalFlops += approximateCountFlops(indexGroup, indexReduced.length > 0, tensorGroup.length, sizeMap);
		const newIndex = indexGroup.filter((x) => !indexReduced.includes(x));
		for (const idx of newIndex) indexUsageCounts[idx]++;
		indices.push(newIndex);
	}
	return totalFlops;
}
function approximateCountFlops(indexGroup, hasReduction, numTerms, sizeMap) {
	return bprod(...indexGroup.map((i) => sizeMap.get(i))) * (BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n));
}
/** Compute size for each index in the einsum expression, also validates input. */
function computeSizeMap({ shapes, lhsIndices, rhsIndex }) {
	if (shapes.length === 0) throw new Error("Einsum must have at least one input tensor");
	if (lhsIndices.length !== shapes.length) throw new Error(`Mismatched number of lhs operands (${lhsIndices.length}) and shapes (${shapes.length})`);
	for (let i = 0; i < shapes.length; i++) if (lhsIndices[i].length !== shapes[i].length) throw new Error(`Mismatched number of indices (${lhsIndices[i].length}) and shape (${JSON.stringify(shapes[i])}) for operand ${i}`);
	const rhsIndexSet = /* @__PURE__ */ new Set();
	for (const idx of rhsIndex) {
		if (rhsIndexSet.has(idx)) throw new Error(`Repeated index ${idx} in einsum output`);
		rhsIndexSet.add(idx);
	}
	const sizeMap = /* @__PURE__ */ new Map();
	for (let i = 0; i < shapes.length; i++) {
		const shape = shapes[i];
		const lhsIndex = lhsIndices[i];
		for (let j = 0; j < lhsIndex.length; j++) {
			const idx = lhsIndex[j];
			const dim = shape[j];
			const existing = sizeMap.get(idx);
			if (existing === void 0 || existing === 1) sizeMap.set(idx, dim);
			else if (existing !== dim && dim !== 1) throw new Error(`Inconsistent size for index ${idx} in einsum: ${existing} vs ${dim}`);
		}
	}
	for (const [idx, size] of sizeMap) if (!Number.isInteger(idx) || idx < 0) throw new Error(`Invalid index ${idx} in einsum expression, must be non-negative integer`);
	else if (size < 0) throw new Error(`Invalid size ${size} for index ${idx} in einsum expression, must be non-negative`);
	for (const idx of rhsIndex) if (!sizeMap.has(idx)) throw new Error(`Output index ${idx} not present in einsum inputs`);
	return sizeMap;
}
const einsumPathCache = /* @__PURE__ */ new Map();
function computeEinsumPath(input, method) {
	if (!method) method = input.shapes.length <= 5 ? "optimal" : "naive";
	return require_backend.runWithCache(einsumPathCache, [input, method], () => {
		const sizeMap = computeSizeMap(input);
		if (input.shapes.length === 1) return new EinsumPath(input, sizeMap, []);
		switch (method) {
			case "naive": return computePathNaive(input, sizeMap);
			case "optimal": return computePathOptimal(input, sizeMap);
			default: throw new Error(`Unknown computePath method: ${method}`);
		}
	});
}
function computePathNaive(input, sizeMap) {
	const n = input.shapes.length;
	const path = [];
	let lastTensorIndex = 0;
	for (let i = 1; i < n; i++) {
		path.push([lastTensorIndex, i]);
		lastTensorIndex = n + i - 1;
	}
	return new EinsumPath(input, sizeMap, path);
}
function computePathOptimal(input, sizeMap) {
	const n = input.shapes.length;
	let bestPath = null;
	let bestFlops = null;
	for (const path of allPaths(require_backend.range(n), n)) {
		const flops = approximatePathFlops(input, sizeMap, path);
		if (bestFlops === null || flops < bestFlops) {
			bestPath = path;
			bestFlops = flops;
		}
	}
	return new EinsumPath(input, sizeMap, bestPath);
}
function* allPaths(tensors, next) {
	if (tensors.length === 2) {
		yield [[tensors[0], tensors[1]]];
		return;
	}
	for (let i = 0; i < tensors.length; i++) for (let j = i + 1; j < tensors.length; j++) {
		const pair = [tensors[i], tensors[j]];
		const newTensors = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
		newTensors.push(next);
		for (const subpath of allPaths(newTensors, next + 1)) yield [pair, ...subpath];
	}
}
//#endregion
//#region src/library/numpy-fft.ts
var numpy_fft_exports = /* @__PURE__ */ __exportAll({
	fft: () => fft,
	fft2: () => fft2,
	fftfreq: () => fftfreq,
	fftn: () => fftn,
	fftshift: () => fftshift,
	hfft: () => hfft,
	ifft: () => ifft,
	ifft2: () => ifft2,
	ifftn: () => ifftn,
	ifftshift: () => ifftshift,
	ihfft: () => ihfft,
	irfft: () => irfft,
	irfft2: () => irfft2,
	irfftn: () => irfftn,
	rfft: () => rfft,
	rfft2: () => rfft2,
	rfftfreq: () => rfftfreq,
	rfftn: () => rfftn
});
function checkPairInput(name, a) {
	const fullName = `jax.numpy.fft.${name}`;
	if (!require_backend.deepEqual(a.real.shape, a.imag.shape)) throw new Error(`${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`);
	if (a.real.dtype !== a.imag.dtype) throw new Error(`${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`);
	if (!require_backend.isFloatDtype(a.real.dtype)) throw new Error(`${fullName}: input must have a float dtype, got ${a.real.dtype}`);
}
function factorFftSize(n) {
	const factors = [];
	for (const radix of [
		4,
		2,
		3,
		5,
		7
	]) while (n % radix === 0) {
		factors.push(radix);
		n /= radix;
	}
	for (let radix = 11; radix * radix <= n; radix += 2) while (n % radix === 0) {
		factors.push(radix);
		n /= radix;
	}
	if (n > 1) factors.push(n);
	return factors;
}
function checkRealInput(name, a) {
	if (!require_backend.isFloatDtype(a.dtype)) throw new Error(`jax.numpy.fft.${name}: input must have a float dtype, got ${a.dtype}`);
}
function checkFrequencyArgs(name, n, d) {
	if (!Number.isInteger(n) || n < 1) throw new Error(`jax.numpy.fft.${name}: n must be a positive integer, got ${n}`);
	if (!Number.isFinite(d) || d === 0) throw new Error(`jax.numpy.fft.${name}: d must be a finite non-zero number, got ${d}`);
}
function sliceAlongAxis(a, axis, start, end) {
	const index = globalThis.Array.from({ length: a.ndim }, () => []);
	index[axis] = [start, end];
	return a.slice(...index);
}
/**
* Compute a one-dimensional discrete Fourier transform.
*/
function fft(a, axis = -1) {
	return fftChecked("fft", a, axis, false);
}
function fftChecked(name, a, axis, inverse) {
	checkPairInput(name, a);
	axis = require_backend.checkAxis(axis, a.real.ndim);
	const n = a.real.shape[axis];
	if (!Number.isInteger(n) || n < 1) throw new Error(`jax.numpy.fft.${name}: size must be a positive integer, got ${n}`);
	return fftRoutine(a, axis, inverse);
}
function fftRoutine(a, axis, inverse) {
	let { real, imag } = a;
	const n = real.shape[axis];
	let perm = null;
	if (axis !== real.ndim - 1) {
		perm = require_backend.range(real.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		real = real.transpose(perm);
		imag = imag.transpose(perm);
	}
	[real, imag] = fft$1(real, imag, {
		factors: factorFftSize(n),
		inverse
	});
	if (perm !== null) {
		real = real.transpose(require_backend.invertPermutation(perm));
		imag = imag.transpose(require_backend.invertPermutation(perm));
	}
	return {
		real,
		imag
	};
}
function transformN(name, a, axes, transform) {
	checkPairInput(name, a);
	const normalizedAxes = require_backend.normalizeAxis(axes, a.real.ndim, false);
	let result = a;
	for (const axis of normalizedAxes) result = transform(result, axis);
	return result;
}
/**
* Compute an N-dimensional discrete Fourier transform.
*/
function fftn(a, axes = null) {
	return transformN("fftn", a, axes, fft);
}
/**
* Compute a two-dimensional discrete Fourier transform.
*/
function fft2(a, axes = [-2, -1]) {
	return fftn(a, axes);
}
/**
* Compute a one-dimensional inverse discrete Fourier transform.
*/
function ifft(a, axis = -1) {
	return fftChecked("ifft", a, axis, true);
}
/**
* Compute an N-dimensional inverse discrete Fourier transform.
*/
function ifftn(a, axes = null) {
	return transformN("ifftn", a, axes, ifft);
}
/**
* Compute a two-dimensional inverse discrete Fourier transform.
*/
function ifft2(a, axes = [-2, -1]) {
	return ifftn(a, axes);
}
/**
* Compute a one-dimensional FFT of real input.
*
* The output stores only the non-negative frequency terms.
*/
function rfft(a, axis = -1) {
	checkRealInput("rfft", a);
	axis = require_backend.checkAxis(axis, a.ndim);
	const n = a.shape[axis];
	const result = fft({
		real: a,
		imag: zerosLike(a.ref)
	}, axis);
	const stop = Math.floor(n / 2) + 1;
	return {
		real: sliceAlongAxis(result.real, axis, 0, stop),
		imag: sliceAlongAxis(result.imag, axis, 0, stop)
	};
}
/**
* Compute the inverse of `rfft`.
*
* The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
* spectrum length.
*/
function irfft(a, axis = -1) {
	checkPairInput("irfft", a);
	const { real, imag } = a;
	axis = require_backend.checkAxis(axis, real.ndim);
	const m = real.shape[axis];
	if (m < 2) throw new Error(`jax.numpy.fft.irfft: packed input length must be at least 2, got ${m}`);
	const mirroredReal = flip(sliceAlongAxis(real.ref, axis, 1, m - 1), axis);
	const mirroredImag = flip(sliceAlongAxis(imag.ref, axis, 1, m - 1), axis).mul(-1);
	const result = ifft({
		real: concatenate([real, mirroredReal], axis),
		imag: concatenate([imag, mirroredImag], axis)
	}, axis);
	result.imag.dispose();
	return result.real;
}
/**
* Compute an N-dimensional FFT of real input.
*
* The final transformed axis stores only the non-negative frequency terms.
*/
function rfftn(a, axes = null) {
	checkRealInput("rfftn", a);
	const normalizedAxes = require_backend.normalizeAxis(axes, a.ndim, false);
	if (normalizedAxes.length === 0) return {
		real: a,
		imag: zerosLike(a.ref)
	};
	const realAxis = normalizedAxes[normalizedAxes.length - 1];
	let result = rfft(a, realAxis);
	for (const axis of normalizedAxes.slice(0, -1)) result = fft(result, axis);
	return result;
}
/**
* Compute a two-dimensional FFT of real input.
*
* The final transformed axis stores only the non-negative frequency terms.
*/
function rfft2(a, axes = [-2, -1]) {
	return rfftn(a, axes);
}
/**
* Compute the inverse of `rfftn`.
*
* The real output length for the final transformed axis is inferred as
* `2 * (m - 1)`.
*/
function irfftn(a, axes = null) {
	checkPairInput("irfftn", a);
	const normalizedAxes = require_backend.normalizeAxis(axes, a.real.ndim, false);
	if (normalizedAxes.length === 0) {
		a.imag.dispose();
		return a.real;
	}
	const realAxis = normalizedAxes[normalizedAxes.length - 1];
	let result = a;
	for (const axis of normalizedAxes.slice(0, -1)) result = ifft(result, axis);
	return irfft(result, realAxis);
}
/**
* Compute the inverse of `rfft2`.
*/
function irfft2(a, axes = [-2, -1]) {
	return irfftn(a, axes);
}
/**
* Compute the FFT of a Hermitian-symmetric signal.
*
* The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
* Hermitian input length.
*/
function hfft(a, axis = -1) {
	checkPairInput("hfft", a);
	axis = require_backend.checkAxis(axis, a.real.ndim);
	const n = 2 * (a.real.shape[axis] - 1);
	return irfft({
		real: a.real,
		imag: a.imag.mul(-1)
	}, axis).mul(n);
}
/** Compute the inverse of `hfft`. */
function ihfft(a, axis = -1) {
	checkRealInput("ihfft", a);
	axis = require_backend.checkAxis(axis, a.ndim);
	const n = a.shape[axis];
	const result = rfft(a, axis);
	return {
		real: result.real.div(n),
		imag: result.imag.mul(-1).div(n)
	};
}
/** Return sample frequencies for a discrete Fourier transform. */
function fftfreq(n, d = 1) {
	checkFrequencyArgs("fftfreq", n, d);
	const scale = 1 / (n * d);
	const positiveEnd = Math.floor((n - 1) / 2) + 1;
	const values = [];
	for (let i = 0; i < positiveEnd; i++) values.push(i * scale);
	for (let i = -Math.floor(n / 2); i < 0; i++) values.push(i * scale);
	return array(values, { dtype: "float32" });
}
/** Return sample frequencies for a real-input discrete Fourier transform. */
function rfftfreq(n, d = 1) {
	checkFrequencyArgs("rfftfreq", n, d);
	const scale = 1 / (n * d);
	const values = [];
	for (let i = 0; i <= Math.floor(n / 2); i++) values.push(i * scale);
	return array(values, { dtype: "float32" });
}
/** Shift the zero-frequency component to the center of the spectrum. */
function fftshift(a, axes = null) {
	const normalizedAxes = require_backend.normalizeAxis(axes, a.ndim, false);
	return roll(a, normalizedAxes.map((axis) => Math.floor(a.shape[axis] / 2)), normalizedAxes);
}
/** Undo `fftshift`. */
function ifftshift(a, axes = null) {
	const normalizedAxes = require_backend.normalizeAxis(axes, a.ndim, false);
	return roll(a, normalizedAxes.map((axis) => -Math.floor(a.shape[axis] / 2)), normalizedAxes);
}
//#endregion
//#region src/library/numpy-linalg.ts
var numpy_linalg_exports = /* @__PURE__ */ __exportAll({
	cholesky: () => cholesky$1,
	cross: () => cross$1,
	det: () => det,
	diagonal: () => diagonal,
	eigh: () => eigh$1,
	eigvalsh: () => eigvalsh,
	inv: () => inv,
	lstsq: () => lstsq,
	matmul: () => matmul,
	matrixNorm: () => matrixNorm,
	matrixPower: () => matrixPower,
	matrixTranspose: () => matrixTranspose,
	multiDot: () => multiDot,
	norm: () => norm,
	outer: () => outer,
	slogdet: () => slogdet,
	solve: () => solve,
	svd: () => svd$1,
	svdvals: () => svdvals,
	tensordot: () => tensordot,
	trace: () => trace,
	vecdot: () => vecdot,
	vectorNorm: () => vectorNorm
});
/**
* Compute the Cholesky decomposition of a (batched) positive-definite matrix.
*
* This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
* the input matrix, which is on by default.
*/
function cholesky$1(a, { upper = false, symmetrizeInput = true } = {}) {
	a = fudgeArray(a);
	require_backend.checkSquare("cholesky", a.shape);
	if (symmetrizeInput) a = a.ref.add(matrixTranspose(a)).mul(.5);
	return cholesky(a, { upper });
}
/**
* Compute the cross-product of two 3D vectors.
*
* This is a simpler and less flexible version of `jax.numpy.cross()`.
* Both inputs must have size 3 along the specified axis.
*/
function cross$1(x1, x2, axis = -1) {
	const a1 = require_backend.checkAxis(axis, ndim(x1));
	const a2 = require_backend.checkAxis(axis, ndim(x2));
	if (shape(x1)[a1] !== 3) throw new Error(`linalg.cross: x1 must have size 3 along axis ${axis}, got ${shape(x1)[a1]}`);
	if (shape(x2)[a2] !== 3) throw new Error(`linalg.cross: x2 must have size 3 along axis ${axis}, got ${shape(x2)[a2]}`);
	return cross(x1, x2, { axis });
}
/** Compute the determinant of a square matrix (batched). */
function det(a) {
	a = fudgeArray(a);
	const n = require_backend.checkSquare("det", a.shape);
	const [lu$2, pivots, permutation] = lu(a);
	permutation.dispose();
	const sign = pivots.notEqual(arange(n)).astype(int32).sum(-1).mod(2).mul(-2).add(1);
	return prod(lu$2.diagonal(0, -1, -2), -1).mul(sign);
}
/**
* Compute eigenvalues and eigenvectors of real symmetric matrices.
*
* The returned eigenvalues are sorted in ascending order, and eigenvectors are
* returned as columns.
*/
function eigh$1(a, opts) {
	const [vectors, values] = eigh(a, { symmetrizeInput: opts?.symmetrizeInput });
	return [values, vectors];
}
/** Compute eigenvalues of real symmetric matrices. */
function eigvalsh(a, opts) {
	const [vectors, values] = eigh(a, { symmetrizeInput: opts?.symmetrizeInput });
	vectors.dispose();
	return values;
}
/**
* Singular value decomposition of real matrices.
*
* This currently computes a thin SVD via symmetric eigendecomposition. It
* supports `fullMatrices=false`; `fullMatrices=true` is accepted only for
* square inputs where thin and full output shapes match.
*/
function svd$1(a, opts) {
	const result = svd(a, {
		fullMatrices: opts?.fullMatrices,
		computeUv: opts?.computeUv
	});
	if (opts?.computeUv === false) return result;
	return result;
}
/** Compute singular values of real matrices. */
function svdvals(a) {
	return svd(a, { computeUv: false });
}
/** Compute the inverse of a square matrix (batched). */
function inv(a) {
	a = fudgeArray(a);
	const n = require_backend.checkSquare("inv", a.shape);
	return solve(a, eye(n, void 0, { dtype: a.dtype }));
}
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
function lstsq(a, b) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	if (a.ndim !== 2) throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
	const [m, n] = a.shape;
	if (b.shape[0] !== m) throw new Error(`lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`);
	const at = matrixTranspose(a.ref);
	if (m <= n) {
		const l = cholesky$1(matmul(a, at.ref), { symmetrizeInput: false });
		return matmul(at, triangularSolve(l, triangularSolve(l.ref, b, {
			leftSide: true,
			lower: true
		}), {
			leftSide: true,
			lower: true,
			transposeA: true
		}));
	} else {
		const l = cholesky$1(matmul(at.ref, a), { symmetrizeInput: false });
		const atb = matmul(at, b);
		return triangularSolve(l, triangularSolve(l.ref, atb, {
			leftSide: true,
			lower: true
		}), {
			leftSide: true,
			lower: true,
			transposeA: true
		});
	}
}
/** Raise a square matrix to an integer power, via repeated squarings. */
function matrixPower(a, n) {
	if (!Number.isInteger(n)) throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
	a = fudgeArray(a);
	const m = require_backend.checkSquare("matrixPower", a.shape);
	if (n === 0) {
		const dtype = a.dtype;
		a.dispose();
		return broadcastTo(eye(m, void 0, { dtype }), a.shape);
	}
	if (n < 0) {
		a = inv(a);
		n = -n;
	}
	let result = null;
	let a2k = a;
	for (let k = 0; n; k++) {
		if (k > 0) a2k = matmul(a2k.ref, a2k);
		if (n % 2 === 1) result = result === null ? a2k.ref : matmul(result, a2k.ref);
		n = Math.floor(n / 2);
	}
	a2k.dispose();
	return result;
}
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
function multiDot(arrays) {
	if (arrays.length < 2) throw new Error("Expected at least two arrays");
	if (arrays[0].ndim === 1) {
		const [first, ...rest] = arrays;
		return multiDot([first.reshape([1, -1]), ...rest]).slice(0);
	}
	if (arrays[arrays.length - 1].ndim === 1) {
		const [rest, last] = [arrays.slice(0, -1), arrays[arrays.length - 1]];
		return multiDot([...rest, last.reshape([-1, 1])]).slice([], 0);
	}
	for (let i = 0; i < arrays.length; i++) if (arrays[i].ndim !== 2) throw new Error(`multiDot: expected 2D, got ${arrays[i].aval} at index ${i}`);
	const einsumArgs = [];
	for (let i = 0; i < arrays.length; i++) einsumArgs.push(arrays[i], [i, i + 1]);
	return einsum(...einsumArgs);
}
/** Return sign and natural logarithm of the determinant of `a`. */
function slogdet(a) {
	a = fudgeArray(a);
	const n = require_backend.checkSquare("slogdet", a.shape);
	const [lu$3, pivots, permutation] = lu(a);
	permutation.dispose();
	let parity = pivots.notEqual(arange(n)).astype(int32).sum(-1);
	const diag = lu$3.diagonal(0, -1, -2);
	parity = parity.add(diag.ref.less(0).astype(int32).sum(-1)).mod(2);
	const logabsdet = log(absolute(diag)).sum(-1);
	return [parity.mul(-2).add(1), logabsdet];
}
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
function solve(a, b) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	const n = require_backend.checkSquare("solve", a.shape);
	if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
	const bIs1d = b.ndim === 1;
	if (bIs1d) b = b.reshape([...b.shape, 1]);
	if (b.shape[b.ndim - 2] !== n) throw new Error(`solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`);
	const m = b.shape[b.ndim - 1];
	const batchDims = require_backend.generalBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2));
	a = broadcastTo(a, [
		...batchDims,
		n,
		n
	]);
	b = broadcastTo(b, [
		...batchDims,
		n,
		m
	]);
	const [lu$4, pivots, permutation] = lu(a);
	pivots.dispose();
	const P = arange(n).equal(permutation.reshape([...permutation.shape, 1])).astype(b.dtype);
	let x = triangularSolve(lu$4, triangularSolve(lu$4.ref, matmul(P, b), {
		leftSide: true,
		lower: true,
		unitDiagonal: true
	}), {
		leftSide: true,
		lower: false
	});
	if (bIs1d) x = squeeze(x, -1);
	return x;
}
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
function norm(x, { ord = null, axis = null, keepdims = false } = {}) {
	x = fudgeArray(x);
	if (!require_backend.isFloatDtype(x.dtype)) x = x.astype(float32);
	const ndim = x.ndim;
	let axes;
	if (axis === null) {
		if (ord === null) return sqrt(sum(square(x), null, { keepdims }));
		axes = require_backend.range(ndim);
	} else if (typeof axis === "number") axes = [require_backend.checkAxis(axis, ndim)];
	else axes = axis.map((a) => require_backend.checkAxis(a, ndim));
	if (axes.length === 1) {
		if (typeof ord === "string") throw new Error(`norm: invalid order '${ord}' for vector norm`);
		return vectorNorm(x, {
			ord: ord ?? 2,
			axis: axes[0],
			keepdims
		});
	} else if (axes.length === 2) {
		let [rowAxis, colAxis] = axes;
		if (rowAxis === colAxis) throw new Error(`norm: duplicate axes ${JSON.stringify(axis)}`);
		if (ord === null || ord === "f" || ord === "fro") return sqrt(sum(square(x), axes, { keepdims }));
		else if (ord === 1 || ord === -1) {
			const sums = sum(absolute(x), rowAxis, { keepdims });
			if (!keepdims && colAxis > rowAxis) colAxis -= 1;
			return ord === 1 ? max(sums, colAxis, { keepdims }) : min(sums, colAxis, { keepdims });
		} else if (ord === Infinity || ord === -Infinity) {
			const sums = sum(absolute(x), colAxis, { keepdims });
			if (!keepdims && rowAxis > colAxis) rowAxis -= 1;
			return ord === Infinity ? max(sums, rowAxis, { keepdims }) : min(sums, rowAxis, { keepdims });
		} else if (ord === 2 || ord === -2 || ord === "nuc") {
			let y = moveaxis(x, rowAxis, -1);
			y = moveaxis(y, colAxis > rowAxis ? colAxis - 1 : colAxis, -1);
			const s = svdvals(y);
			const result = ord === 2 ? max(s, -1) : ord === -2 ? min(s, -1) : sum(s, -1);
			return keepdims ? expandDims(result, axes) : result;
		} else throw new Error(`norm: invalid order '${ord}' for matrix norm`);
	} else throw new Error(`norm: axis must be null, an integer, or a pair of integers, got ${JSON.stringify(axis)}`);
}
/**
* Compute the vector norm of an array.
*
* @param x - Input array.
* @param ord - Order of the norm (default 2). Supports `Infinity`, `-Infinity`, `0`, or any real number.
* @param axis - Axis/axes to reduce over (default: all axes).
* @param keepdims - Whether to keep reduced dimensions as size 1.
* @returns The norm of `x`, reduced over the given axes.
*/
function vectorNorm(x, { ord = 2, axis = null, keepdims = false } = {}) {
	x = fudgeArray(x);
	const ax = axis ?? null;
	if (ord === Infinity) return max(absolute(x), ax, { keepdims });
	else if (ord === -Infinity) return min(absolute(x), ax, { keepdims });
	else if (ord === 0) return x.notEqual(0).astype(x.dtype).sum(ax, { keepdims });
	else return power(power(absolute(x), ord).sum(ax, { keepdims }), 1 / ord);
}
/**
* Calculates the matrix norm of a matrix or a stack of matrices.
* @param x - Input Array (..., M, N)
* @param ord - Order of the norm (default "fro").
* @param keepdims - Whether to keep reduced dimensions as size 1 or collapse axes.
* @returns Matrix norm of input array. If `keepdims` is `true`, the result has shape (..., 1, 1). Otherwise, it has shape (...,).
*/
function matrixNorm(x, { ord = "fro", keepdims = false } = {}) {
	x = fudgeArray(x);
	if (x.ndim < 2) throw new Error(`Input must be at least 2-dimensional. Recieved ${x.ndim}-dimensional array`);
	if (ord === "fro") return sqrt(sum(square(x), [-2, -1], { keepdims }));
	else if (ord === "nuc") throw new Error("Order nuc is not supported");
	else if (ord === -1) return min(sum(absolute(x), [-2], { keepdims }), [-1], { keepdims });
	else if (ord === 1) return max(sum(absolute(x), [-2], { keepdims }), [-1], { keepdims });
	else if (ord === -2) throw new Error("Order -2 is not supported");
	else if (ord === 2) throw new Error("Order 2 is not supported");
	else if (ord === -Infinity) return min(sum(absolute(x), [-1], { keepdims }), [keepdims ? -2 : -1], { keepdims });
	else if (ord === Infinity) return max(sum(absolute(x), [-1], { keepdims }), [keepdims ? -2 : -1], { keepdims });
	else throw new Error(`Order ${ord} is not supported`);
}
//#endregion
//#region src/library/numpy/dtype-info.ts
/** Machine limits for floating-point types. */
function finfo(dtype) {
	if (!require_backend.isFloatDtype(dtype)) throw new Error(`finfo: received ${dtype}, must be a floating-point type`);
	switch (dtype) {
		case "float16": return Object.freeze({
			bits: 16,
			dtype: "float16",
			eps: 2 ** -10,
			epsneg: 2 ** -11,
			machep: -10,
			max: 65504,
			maxexp: 16,
			min: -65504,
			minexp: -14,
			negep: -24,
			nexp: 5,
			nmant: 10,
			precision: 3,
			resolution: .001,
			smallestNormal: 2 ** -14,
			smallestSubnormal: 2 ** -24
		});
		case "float32": return Object.freeze({
			bits: 32,
			dtype: "float32",
			eps: 2 ** -23,
			epsneg: 2 ** -24,
			machep: -23,
			max: 34028234663852886e22,
			maxexp: 128,
			min: -34028234663852886e22,
			minexp: -126,
			negep: -24,
			nexp: 8,
			nmant: 23,
			precision: 6,
			resolution: 1e-6,
			smallestNormal: 2 ** -126,
			smallestSubnormal: 2 ** -149
		});
		case "float64": return Object.freeze({
			bits: 64,
			dtype: "float64",
			eps: 2 ** -52,
			epsneg: 2 ** -53,
			machep: -52,
			max: Number.MAX_VALUE,
			maxexp: 1024,
			min: -Number.MAX_VALUE,
			minexp: -1022,
			negep: -53,
			nexp: 11,
			nmant: 52,
			precision: 15,
			resolution: 1e-15,
			smallestNormal: 2 ** -1022,
			smallestSubnormal: 2 ** -1074
		});
		default: throw new Error(`finfo: unsupported dtype ${dtype}`);
	}
}
/** Machine limits for integer types. */
function iinfo(dtype) {
	switch (dtype) {
		case "int32": return Object.freeze({
			bits: 32,
			dtype: "int32",
			max: 2147483647,
			min: -2147483648
		});
		case "uint32": return Object.freeze({
			bits: 32,
			dtype: "uint32",
			max: 4294967295,
			min: 0
		});
		default: throw new Error(`iinfo: unsupported dtype ${dtype}`);
	}
}
//#endregion
//#region src/library/numpy.ts
var numpy_exports = /* @__PURE__ */ __exportAll({
	Array: () => Array$1,
	DType: () => require_backend.DType,
	abs: () => absolute,
	absolute: () => absolute,
	acos: () => acos,
	acosh: () => arccosh,
	add: () => add,
	all: () => all,
	allclose: () => allclose,
	any: () => any,
	append: () => append,
	applyAlongAxis: () => applyAlongAxis,
	applyOverAxes: () => applyOverAxes,
	arange: () => arange,
	arccos: () => acos,
	arccosh: () => arccosh,
	arcsin: () => asin,
	arcsinh: () => arcsinh,
	arctan: () => atan,
	arctan2: () => atan2,
	arctanh: () => arctanh,
	argmax: () => argmax,
	argmin: () => argmin,
	argsort: () => argsort,
	around: () => round,
	array: () => array,
	arrayEqual: () => arrayEqual,
	arrayEquiv: () => arrayEquiv,
	arraySplit: () => arraySplit,
	asin: () => asin,
	asinh: () => arcsinh,
	astype: () => astype,
	atan: () => atan,
	atan2: () => atan2,
	atanh: () => arctanh,
	average: () => average,
	bartlett: () => bartlett,
	bitwiseAnd: () => bitwiseAnd,
	bitwiseCount: () => bitwiseCount,
	bitwiseInvert: () => invert,
	bitwiseLeftShift: () => leftShift,
	bitwiseNot: () => invert,
	bitwiseOr: () => bitwiseOr,
	bitwiseRightShift: () => rightShift,
	bitwiseXor: () => bitwiseXor,
	blackman: () => blackman,
	bool: () => bool,
	broadcastArrays: () => broadcastArrays,
	broadcastShapes: () => broadcastShapes,
	broadcastTo: () => broadcastTo,
	cbrt: () => cbrt,
	ceil: () => ceil,
	clip: () => clip,
	columnStack: () => columnStack,
	concatenate: () => concatenate,
	convolve: () => convolve,
	copysign: () => copysign,
	corrcoef: () => corrcoef,
	correlate: () => correlate,
	cos: () => cos,
	cosh: () => cosh,
	countNonzero: () => countNonzero,
	cov: () => cov,
	cross: () => cross,
	cumprod: () => cumprod,
	cumsum: () => cumsum,
	cumulativeProd: () => cumulativeProd,
	cumulativeSum: () => cumulativeSum,
	deg2rad: () => deg2rad,
	degrees: () => degrees,
	diag: () => diag,
	diagIndices: () => diagIndices,
	diagIndicesFrom: () => diagIndicesFrom,
	diagflat: () => diagflat,
	diagonal: () => diagonal,
	diff: () => diff,
	divide: () => trueDivide,
	divmod: () => divmod,
	dot: () => dot$1,
	dsplit: () => dsplit,
	dstack: () => dstack,
	e: () => e,
	ediff1d: () => ediff1d,
	einsum: () => einsum,
	equal: () => equal,
	eulerGamma: () => eulerGamma,
	exp: () => exp,
	exp2: () => exp2,
	expandDims: () => expandDims,
	expm1: () => expm1,
	eye: () => eye,
	fft: () => numpy_fft_exports,
	finfo: () => finfo,
	flip: () => flip,
	fliplr: () => fliplr,
	flipud: () => flipud,
	float16: () => float16,
	float32: () => float32,
	float64: () => float64,
	floatPower: () => floatPower,
	floor: () => floor,
	floorDivide: () => floorDivide,
	fmax: () => fmax,
	fmin: () => fmin,
	fmod: () => fmod,
	frexp: () => frexp,
	fromfunction: () => fromfunction,
	full: () => full,
	fullLike: () => fullLike,
	geomspace: () => geomspace,
	greater: () => greater,
	greaterEqual: () => greaterEqual,
	hann: () => hann,
	heaviside: () => heaviside,
	hsplit: () => hsplit,
	hstack: () => hstack,
	hypot: () => hypot,
	identity: () => identity$1,
	iinfo: () => iinfo,
	indices: () => indices,
	inf: () => inf,
	inner: () => inner,
	int32: () => int32,
	invert: () => invert,
	isfinite: () => isfinite,
	isin: () => isin,
	isinf: () => isinf,
	isnan: () => isnan,
	isneginf: () => isneginf,
	isposinf: () => isposinf,
	isscalar: () => isscalar,
	ldexp: () => ldexp,
	leftShift: () => leftShift,
	less: () => less,
	lessEqual: () => lessEqual,
	linalg: () => numpy_linalg_exports,
	linspace: () => linspace,
	log: () => log,
	log10: () => log10,
	log1p: () => log1p,
	log2: () => log2,
	logaddexp: () => logaddexp,
	logaddexp2: () => logaddexp2,
	logicalAnd: () => logicalAnd,
	logicalNot: () => logicalNot,
	logicalOr: () => logicalOr,
	logicalXor: () => logicalXor,
	logspace: () => logspace,
	matmul: () => matmul,
	matrixTranspose: () => matrixTranspose,
	matvec: () => matvec,
	max: () => max,
	maximum: () => maximum,
	mean: () => mean,
	meshgrid: () => meshgrid,
	min: () => min,
	minimum: () => minimum,
	modf: () => modf,
	moveaxis: () => moveaxis,
	multiply: () => multiply,
	nan: () => nan,
	nanToNum: () => nanToNum,
	nanargmax: () => nanargmax,
	nanargmin: () => nanargmin,
	nancumprod: () => nancumprod,
	nancumsum: () => nancumsum,
	nanmax: () => nanmax,
	nanmean: () => nanmean,
	nanmin: () => nanmin,
	nanprod: () => nanprod,
	nanstd: () => nanstd,
	nansum: () => nansum,
	nanvar: () => nanvar,
	ndim: () => ndim,
	negative: () => negative,
	notEqual: () => notEqual,
	ones: () => ones,
	onesLike: () => onesLike,
	outer: () => outer,
	pad: () => pad,
	permuteDims: () => transpose,
	pi: () => pi,
	polyadd: () => polyadd,
	polyder: () => polyder,
	polymul: () => polymul,
	polysub: () => polysub,
	polyval: () => polyval,
	positive: () => positive,
	pow: () => power,
	power: () => power,
	prod: () => prod,
	promoteTypes: () => require_backend.promoteTypes,
	ptp: () => ptp,
	rad2deg: () => rad2deg,
	radians: () => radians,
	ravel: () => ravel,
	reciprocal: () => reciprocal,
	remainder: () => remainder,
	repeat: () => repeat,
	reshape: () => reshape,
	resultType: () => resultType,
	rightShift: () => rightShift,
	rint: () => rint,
	roll: () => roll,
	rot90: () => rot90,
	round: () => round,
	select: () => select,
	shape: () => shape,
	sign: () => sign,
	signbit: () => signbit,
	sin: () => sin,
	sinc: () => sinc,
	sinh: () => sinh,
	size: () => size,
	sort: () => sort,
	split: () => split$1,
	sqrt: () => sqrt,
	square: () => square,
	squeeze: () => squeeze,
	stack: () => stack,
	std: () => std,
	subtract: () => subtract,
	sum: () => sum,
	swapaxes: () => swapaxes,
	take: () => take,
	takeAlongAxis: () => takeAlongAxis,
	tan: () => tan,
	tanh: () => tanh,
	tensordot: () => tensordot,
	tile: () => tile,
	trace: () => trace,
	transpose: () => transpose,
	trapezoid: () => trapezoid,
	tri: () => tri,
	tril: () => tril,
	triu: () => triu,
	trueDivide: () => trueDivide,
	trunc: () => trunc,
	uint32: () => uint32,
	unstack: () => unstack,
	unwrap: () => unwrap,
	vander: () => vander,
	var_: () => var_,
	vdot: () => vdot,
	vecdot: () => vecdot,
	vecmat: () => vecmat,
	vsplit: () => vsplit,
	vstack: () => vstack,
	where: () => where,
	zeros: () => zeros,
	zerosLike: () => zerosLike
});
const JsArray$2 = globalThis.Array;
const float32 = "float32";
const int32 = "int32";
const uint32 = "uint32";
const bool = "bool";
const float16 = "float16";
const float64 = "float64";
/** Euler's constant, `e = 2.7182818284590...` */
const e = Math.E;
/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
const eulerGamma = .5772156649015329;
/** Positive infinity. */
const inf = Number.POSITIVE_INFINITY;
/** Floating-point representation of NaN. */
const nan = NaN;
/** This is Pi, `π = 3.14159265358979...` */
const pi = Math.PI;
/** @function Element-wise addition, with broadcasting. */
const add = add$1;
/** @function Element-wise multiplication, with broadcasting. */
const multiply = mul;
/** @function Numerical negative of every element of an array. */
const negative = neg;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
const reciprocal = reciprocal$1;
/** @function Round input down to the nearest integer. */
const floor = floor$1;
/** @function Round input up to the nearest integer. */
const ceil = ceil$1;
/** @function Element-wise sine function (takes radians). */
const sin = sin$1;
/** @function Element-wise cosine function (takes radians). */
const cos = cos$1;
/** @function Element-wise inverse sine function (inverse of sin). */
const asin = asin$1;
/** @function Element-wise inverse tangent function (inverse of tan). */
const atan = atan$1;
/** @function Calculate the exponential of all elements in the input array. */
const exp = exp$1;
/** @function Calculate the natural logarithm of all elements in the input array. */
const log = log$1;
/** @function Calculate the square root of all elements in the input array. */
const sqrt = sqrt$1;
/** @function Return element-wise minimum of the input arrays. */
const minimum = min$1;
/** @function Return element-wise maximum of the input arrays. */
const maximum = max$1;
/** @function Compare two arrays element-wise. */
const greater = greater$1;
/** @function Compare two arrays element-wise. */
const less = less$1;
/** @function Compare two arrays element-wise. */
const equal = equal$1;
/** @function Compare two arrays element-wise. */
const notEqual = notEqual$1;
/** @function Compare two arrays element-wise. */
const greaterEqual = greaterEqual$1;
/** @function Compare two arrays element-wise. */
const lessEqual = lessEqual$1;
/** Compute element-wise logical AND. */
function logicalAnd(x, y) {
	return astype(x, bool).mul(astype(y, bool));
}
/** Compute element-wise logical OR. */
function logicalOr(x, y) {
	return astype(x, bool).add(astype(y, bool));
}
/** Compute element-wise logical XOR. */
function logicalXor(x, y) {
	return notEqual(astype(x, bool), astype(y, bool));
}
/** Compute element-wise logical NOT. */
function logicalNot(x) {
	return notEqual(astype(x, bool), true);
}
/** Compute element-wise bitwise AND. */
function bitwiseAnd(x, y) {
	return bitCombine(x, y, "and");
}
/** Compute element-wise bitwise OR. */
function bitwiseOr(x, y) {
	return bitCombine(x, y, "or");
}
/** Compute element-wise bitwise XOR. */
function bitwiseXor(x, y) {
	return bitCombine(x, y, "xor");
}
/**
* Count the number of 1 bits in the binary representation of the absolute
* value of each element.
*/
function bitwiseCount(x) {
	const arr = fudgeArray(x);
	switch (arr.dtype) {
		case "bool": return astype(arr, int32);
		case "uint32": return bitCount(arr);
		case "int32": return bitCount(absolute(arr));
		default: throw new TypeError(`bitwiseCount: unsupported dtype ${arr.dtype}`);
	}
}
/** Compute element-wise bitwise NOT (inversion). */
function invert(x) {
	const arr = fudgeArray(x);
	let allOnes;
	switch (arr.dtype) {
		case "bool":
			allOnes = true;
			break;
		case "uint32":
			allOnes = 4294967295;
			break;
		case "int32":
			allOnes = -1;
			break;
		default: throw new TypeError(`invert: unsupported dtype ${arr.dtype}`);
	}
	return bitCombine(arr, allOnes, "xor");
}
/** Compute element-wise left bit shift. */
function leftShift(x, y) {
	return bitShift(x, y, "shl");
}
/** Compute element-wise right bit shift. */
function rightShift(x, y) {
	return bitShift(x, y, "shr");
}
/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
const where = where$1;
/**
* @function
* Permute the dimensions of an array. Defaults to reversing the axis order.
*/
const transpose = transpose$1;
/**
* @function
* Give a new shape to an array without changing its data.
*
* One shape dimension can be -1. In this case, the value is inferred from the
* length of the array and remaining dimensions.
*/
const reshape = reshape$1;
/**
* @function
* Move axes of an array to new positions. Other axes retain original order.
*/
const moveaxis = moveaxis$1;
/**
* @function
* Add padding (zeros) to an array.
*
* The `width` argument is either an integer or pair of integers, in which case
* all axes are padded with the same width. Or if it is an array of pairs, each
* pair specifies the padding for its corresponding axis.
*/
const pad = pad$1;
/**
* @function
* Return the number of dimensions of an array. Does not consume array reference.
*/
const ndim = ndim$1;
/** @function Return the shape of an array. Does not consume array reference. */
const shape = getShape;
/**
* @function
* Return an array of zeros with the same shape and type as a given array.
*/
const zerosLike = zerosLike$1;
/**
* @function
* Return an array of ones with the same shape and type as a given array.
*/
const onesLike = onesLike$1;
/**
* @function
* Return a full array with the same shape and type as a given array.
*/
const fullLike = fullLike$1;
/**
* Return the number of elements in an array, optionally along an axis.
* Does not consume array reference.
*/
function size(a, axis) {
	const s = shape(a);
	return axis === void 0 ? require_backend.prod(s) : s[axis];
}
/**
* Return true if the input is a scalar.
*
* Like JAX (and unlike NumPy), this treats zero-dimensional arrays as
* scalars. Returns true for numbers, booleans, and 0-dimensional arrays,
* and false for any other value. Does not consume array reference.
*/
function isscalar(element) {
	if (element instanceof Tracer) return element.ndim === 0;
	return typeof element === "number" || typeof element === "boolean";
}
/** Convert an array to a specified dtype. */
function astype(a, dtype) {
	return fudgeArray(a).astype(dtype);
}
/**
* Return the dtype resulting from applying type promotion rules to the
* arguments, which may be arrays, JS numbers or booleans, or dtypes.
*
* Weakly typed values (e.g., JS numbers) only affect the result when every
* argument is weakly typed, following the same rules as `promoteTypes()`.
* Does not consume array references.
*/
function resultType(...args) {
	if (args.length === 0) throw new TypeError("resultType requires at least one argument");
	return args.map((x) => {
		if (typeof x === "string") {
			if (!Object.values(require_backend.DType).includes(x)) throw new TypeError(`resultType: invalid dtype '${x}'`);
			return new ShapedArray([], x, false);
		}
		const { dtype, weakType } = getAval(x);
		return new ShapedArray([], dtype, weakType);
	}).reduce((a, b) => promoteAvals(a, b)).dtype;
}
/** Sum of the elements of the array over a given axis, or axes. */
function sum(a, axis = null, opts) {
	return reduce(a, "Add", axis, opts);
}
/**
* Sum of the elements of the array over a given axis, or axes, treating NaNs
* as zero.
*/
function nansum(a, axis = null, opts) {
	a = fudgeArray(a);
	if (require_backend.isFloatDtype(a.dtype)) a = where(isnan(a.ref), 0, a);
	return sum(a, axis, opts);
}
/** Count the number of non-zero elements along the specified axis. */
function countNonzero(a, axis = null, opts) {
	return sum(astype(notEqual(a, 0), int32), axis, opts);
}
/** Product of the array elements over a given axis. */
function prod(a, axis = null, opts) {
	return reduce(a, "Mul", axis, opts);
}
/** Product of the array elements over a given axis, treating NaNs as one. */
function nanprod(a, axis = null, opts) {
	a = fudgeArray(a);
	if (require_backend.isFloatDtype(a.dtype)) a = where(isnan(a.ref), 1, a);
	return prod(a, axis, opts);
}
/** Return the minimum of array elements along a given axis. */
function min(a, axis = null, opts) {
	return reduce(a, "Min", axis, opts);
}
/** Return the maximum of array elements along a given axis. */
function max(a, axis = null, opts) {
	return reduce(a, "Max", axis, opts);
}
/**
* Return the maximum of array elements along a given axis, ignoring NaNs.
*
* Slices that contain only NaN values return NaN.
*/
function nanmax(a, axis = null, opts) {
	a = fudgeArray(a);
	axis = require_backend.normalizeAxis(axis, a.ndim);
	if (axis.some((i) => a.shape[i] === 0)) throw new Error("zero-size array to reduction operation max which has no identity");
	if (!require_backend.isFloatDtype(a.dtype)) return max(a, axis, opts);
	const mask = isnan(a.ref);
	const out = max(where(mask.ref, -Infinity, a), axis, opts);
	return where(all(mask, axis, opts), NaN, out);
}
/**
* Test whether any array element along a given axis evaluates to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function any(a, axis = null, opts) {
	return fudgeArray(a).any(axis, opts);
}
/**
* Test whether all array elements along a given axis evaluate to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function all(a, axis = null, opts) {
	return fudgeArray(a).all(axis, opts);
}
/** Return the peak-to-peak range along a given axis (`max - min`). */
function ptp(a, axis = null, opts) {
	a = fudgeArray(a);
	return max(a.ref, axis, opts).sub(min(a, axis, opts));
}
/** Compute the average of the array elements along the specified axis. */
function mean(a, axis = null, opts) {
	return fudgeArray(a).mean(axis, opts);
}
/**
* Compute the average of the array elements along the specified axis, ignoring
* NaNs. Slices that are all-NaN produce NaN.
*/
function nanmean(a, axis = null, opts) {
	a = fudgeArray(a);
	if (!require_backend.isFloatDtype(a.dtype)) return mean(a, axis, opts);
	const normalizer = sum(astype(logicalNot(isnan(a.ref)), a.dtype), axis, opts);
	return nansum(a, axis, opts).div(normalizer);
}
/**
* Compute the weighted average along the specified axis.
*
* If no axis is specified, mean is computed along all the axes. The weights
* should have shape matching that of `a`, or if an axis is specified, it should
* match the shape along those axes.
*/
function average(a, axis = null, opts) {
	a = fudgeArray(a);
	if (opts?.weights == null) return mean(a, axis, opts);
	const weights = fudgeArray(opts.weights);
	axis = require_backend.normalizeAxis(axis, ndim(a));
	const wShape = weights.shape;
	const aShape = a.shape;
	if (require_backend.deepEqual(wShape, aShape)) {
		const scl = sum(weights.ref, axis, opts);
		return sum(multiply(a, weights), axis, opts).div(scl);
	} else if (axis.length === 1 && wShape.length === 1 && wShape[0] === aShape[axis[0]]) {
		const wReshaped = reshape(weights, aShape.map((_, i) => i === axis[0] ? wShape[0] : 1));
		const scl = sum(wReshaped.ref, axis, opts);
		return sum(multiply(a, wReshaped), axis, opts).div(scl);
	} else {
		weights.dispose();
		a.dispose();
		throw new Error(`average: weights shape ${JSON.stringify(wShape)} is not compatible with array shape ${JSON.stringify(aShape)} and axis ${JSON.stringify(axis)}`);
	}
}
/**
* Returns the indices of the minimum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmin(a, axis, opts) {
	a = fudgeArray(a);
	let flattenedNdim;
	if (axis === void 0) {
		flattenedNdim = a.ndim;
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const shape = a.shape;
	if (shape[axis] === 0) throw new Error("attempt to get argmin of an empty sequence");
	const isMax = equal(a, min(a.ref, axis, { keepdims: true }));
	const length = array(shape[axis], {
		dtype: int32,
		device: a.device
	});
	const idx = isMax.astype("int32").mul(arange(shape[axis], 0, -1, {
		dtype: int32,
		device: a.device
	}).reshape([shape[axis], ...require_backend.rep(shape.length - axis - 1, 1)]));
	const result = length.sub(max(idx, axis, opts));
	return flattenedNdim !== void 0 && opts?.keepdims ? result.reshape(require_backend.rep(flattenedNdim, 1)) : result;
}
/**
* Returns the indices of the minimum values along an axis, ignoring NaNs.
*
* By default, index is into the flattened array, otherwise it is along the
* specified axis. Unlike NumPy, which raises an error for all-NaN slices, this
* returns -1 for those slices, matching JAX.
*/
function nanargmin(a, axis, opts) {
	a = fudgeArray(a);
	if (!require_backend.isFloatDtype(a.dtype)) return argmin(a, axis, opts);
	const nanMask = isnan(a.ref);
	const result = argmin(where(nanMask.ref, Infinity, a), axis, opts);
	return where(all(nanMask, axis ?? null, opts), -1, result);
}
/**
* Returns the indices of the maximum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmax(a, axis, opts) {
	a = fudgeArray(a);
	let flattenedNdim;
	if (axis === void 0) {
		flattenedNdim = a.ndim;
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const shape = a.shape;
	if (shape[axis] === 0) throw new Error("attempt to get argmax of an empty sequence");
	const isMax = equal(a, max(a.ref, axis, { keepdims: true }));
	const length = array(shape[axis], {
		dtype: int32,
		device: a.device
	});
	const idx = isMax.astype("int32").mul(arange(shape[axis], 0, -1, {
		dtype: int32,
		device: a.device
	}).reshape([shape[axis], ...require_backend.rep(shape.length - axis - 1, 1)]));
	const result = length.sub(max(idx, axis, opts));
	return flattenedNdim !== void 0 && opts?.keepdims ? result.reshape(require_backend.rep(flattenedNdim, 1)) : result;
}
/**
* Returns the indices of the maximum values along an axis, ignoring NaNs.
*
* By default, index is into the flattened array, otherwise it is along the
* specified axis. Unlike NumPy, which raises an error for all-NaN slices, this
* returns -1 for those slices, matching JAX.
*/
function nanargmax(a, axis, opts) {
	a = fudgeArray(a);
	if (!require_backend.isFloatDtype(a.dtype)) return argmax(a, axis, opts);
	const nanMask = isnan(a.ref);
	const result = argmax(where(nanMask.ref, -Infinity, a), axis, opts);
	return where(all(nanMask, axis ?? null, opts), -1, result);
}
/**
* Direct O(n^2) cumulative reduction of the last axis, by a masked reduction.
* The diagonal offset `k` selects an inclusive (0) or exclusive (-1) scan.
*/
function cumulativeQuadratic(op, a, padValue, k = 0) {
	const n = a.shape[a.ndim - 1];
	a = broadcast(a, a.shape.concat(n), [-2]);
	return reduce(where(tri(n, n, k, { dtype: bool }), a, padValue), op, -1);
}
function cumulativeHelper(op, a, axis) {
	const padValue = op === "Add" ? 0 : 1;
	a = fudgeArray(a);
	if (a.ndim === 0) a = a.reshape([1]);
	axis = require_backend.checkAxis(axis, a.ndim);
	if (a.size === 0) return a;
	a = moveaxis(a, axis, -1);
	const n = a.shape[a.ndim - 1];
	const split = 256;
	if (n <= split * 2) return moveaxis(cumulativeQuadratic(op, a, padValue), -1, axis);
	const batch = a.shape.slice(0, -1);
	const cols = batch.map(() => []);
	const m = Math.ceil(n / split);
	a = pad$1(a, { [a.ndim - 1]: [0, m * split - n] });
	const scans = cumulativeQuadratic(op, a.reshape([
		...batch,
		m,
		split
	]), padValue);
	const offsets = cumulativeQuadratic(op, scans.ref.slice(...cols, [], -1), padValue, -1);
	return moveaxis((op === "Add" ? scans.add(offsets.reshape([
		...batch,
		m,
		1
	])) : scans.mul(offsets.reshape([
		...batch,
		m,
		1
	]))).reshape([...batch, m * split]).slice(...cols, [0, n]), -1, axis);
}
/** Cumulative sum of elements along an axis. */
function cumsum(a, axis) {
	if (axis === void 0) a = ravel(a), axis = 0;
	return cumulativeHelper("Add", a, axis);
}
/** Alternative API for `jax.numpy.cumsum()`. */
function cumulativeSum(x, opts) {
	x = fudgeArray(x);
	if (x.ndim === 0) x = x.reshape([1]);
	if (x.ndim > 1 && opts?.axis == void 0) throw new Error("cumulativeSum: axis is required for arrays of ndim > 1");
	const axis = require_backend.checkAxis(opts?.axis ?? 0, x.ndim);
	if (opts?.includeInitial) x = concatenate([zerosLike(x.ref, { shape: x.shape.toSpliced(axis, 1, 1) }), x], axis);
	return cumulativeHelper("Add", x, axis);
}
/** Cumulative product of elements along an axis. */
function cumprod(a, axis) {
	if (axis === void 0) a = ravel(a), axis = 0;
	return cumulativeHelper("Mul", a, axis);
}
/** Alternative API for `jax.numpy.cumprod()`. */
function cumulativeProd(x, opts) {
	x = fudgeArray(x);
	if (x.ndim === 0) x = x.reshape([1]);
	if (x.ndim > 1 && opts?.axis == void 0) throw new Error("cumulativeProd: axis is required for arrays of ndim > 1");
	const axis = require_backend.checkAxis(opts?.axis ?? 0, x.ndim);
	if (opts?.includeInitial) x = concatenate([onesLike(x.ref, { shape: x.shape.toSpliced(axis, 1, 1) }), x], axis);
	return cumulativeHelper("Mul", x, axis);
}
/** Cumulative product of elements along an axis, treating NaNs as one. */
function nancumprod(a, axis) {
	a = fudgeArray(a);
	if (require_backend.isFloatDtype(a.dtype)) a = where(isnan(a.ref), 1, a);
	return cumprod(a, axis);
}
/** Cumulative sum of elements along an axis, treating NaNs as zero. */
function nancumsum(a, axis) {
	a = fudgeArray(a);
	if (require_backend.isFloatDtype(a.dtype)) a = where(isnan(a.ref), 0, a);
	return cumsum(a, axis);
}
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
function trapezoid(y, x = null, opts) {
	y = fudgeArray(y);
	const requestedAxis = opts?.axis ?? -1;
	const axis = require_backend.checkAxis(requestedAxis, y.ndim);
	const sliceAxis = (a, ax, s) => a.slice(...require_backend.rep(ax, []), s);
	let dx;
	if (x === null) {
		if (!require_backend.isFloatDtype(y.dtype)) y = y.astype(float32);
		dx = fudgeArray(opts?.dx ?? 1);
		if (dx.ndim > 0) {
			if (dx.ndim < y.ndim) dx = dx.reshape(require_backend.rep(y.ndim - dx.ndim, 1).concat(dx.shape));
			dx = moveaxis(dx, requestedAxis, -1);
		}
	} else {
		x = fudgeArray(x);
		let dtype = require_backend.promoteTypes(y.dtype, x.dtype);
		if (!require_backend.isFloatDtype(dtype)) dtype = float32;
		if (y.dtype !== dtype) y = y.astype(dtype);
		if (x.dtype !== dtype) x = x.astype(dtype);
		const xAxis = x.ndim === 1 ? 0 : require_backend.checkAxis(requestedAxis, x.ndim);
		const diff = sliceAxis(x.ref, xAxis, [1]).sub(sliceAxis(x, xAxis, [0, -1]));
		dx = x.ndim === 1 ? diff : moveaxis(diff, xAxis, -1);
	}
	y = moveaxis(y, axis, -1);
	return sliceAxis(y.ref, y.ndim - 1, [1]).add(sliceAxis(y, y.ndim - 1, [0, -1])).mul(.5).mul(dx).sum(-1);
}
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
function diff(a, n = 1, axis = -1, opts) {
	a = fudgeArray(a);
	if (!Number.isInteger(n) || n < 0) {
		a.dispose();
		throw new Error(`diff: order must be a non-negative integer, got ${n}`);
	}
	if (n === 0) return a;
	if (a.ndim === 0) {
		a.dispose();
		throw new Error("diff: input must be at least one-dimensional");
	}
	axis = require_backend.checkAxis(axis, a.ndim);
	if (opts?.prepend !== void 0 || opts?.append !== void 0) {
		const edgeShape = a.shape.toSpliced(axis, 1, 1);
		const combined = [a];
		if (opts.prepend !== void 0) {
			let prepend = fudgeArray(opts.prepend);
			if (prepend.ndim === 0) prepend = broadcastTo(prepend, edgeShape);
			combined.unshift(prepend);
		}
		if (opts.append !== void 0) {
			let append = fudgeArray(opts.append);
			if (append.ndim === 0) append = broadcastTo(append, edgeShape);
			combined.push(append);
		}
		a = concatenate(combined, axis);
	}
	const op = a.dtype === bool ? notEqual : subtract;
	const upper = a.shape.map((_, i) => i === axis ? [1] : []);
	const lower = a.shape.map((_, i) => i === axis ? [0, -1] : []);
	for (let i = Math.min(n, a.shape[axis]); i > 0; i--) a = op(a.ref.slice(...upper), a.slice(...lower));
	return a;
}
/**
* Compute the differences between consecutive elements of an array.
*
* The input is flattened before taking differences. `opts.toBegin` and
* `opts.toEnd` are flattened, cast to the input dtype, and prepended or
* appended to the result.
*/
function ediff1d(ary, opts) {
	const arr = ravel(ary);
	const n = arr.size;
	const dtype = arr.dtype;
	const parts = [arr.ref.slice([Math.min(n, 1)]).sub(arr.slice([0, Math.max(n - 1, 0)]))];
	if (opts?.toBegin !== void 0) parts.unshift(ravel(opts.toBegin).astype(dtype));
	if (opts?.toEnd !== void 0) parts.push(ravel(opts.toEnd).astype(dtype));
	return parts.length === 1 ? parts[0] : concatenate(parts, 0);
}
/** Reverse the elements in an array along the given axes. */
function flip(x, axis = null) {
	const nd = ndim(x);
	axis = require_backend.normalizeAxis(axis, nd);
	return flip$1(x, axis);
}
/**
* Split an array into multiple sub-arrays along an axis.
*
* @param a - The input array to split.
* @param indicesOrSections - If an integer, it indicates the number of equal
* sections to create along the specified axis. If a list of integers, it
* specifies the indices at which to split the array.
* @param axis - The axis along which to split the array. Default is 0.
*/
function split$1(a, indicesOrSections, axis = 0) {
	a = fudgeArray(a);
	axis = require_backend.checkAxis(axis, a.ndim);
	const size = a.shape[axis];
	let sizes;
	if (typeof indicesOrSections === "number") {
		if (size % indicesOrSections !== 0) throw new Error(`Array of size ${size} cannot be split into ${indicesOrSections} equal parts`);
		sizes = require_backend.rep(indicesOrSections, size / indicesOrSections);
	} else {
		const bounds = [
			0,
			...indicesOrSections.map((i) => i < 0 ? i + size : i),
			size
		];
		sizes = require_backend.range(bounds.length - 1).map((i) => bounds[i + 1] - bounds[i]);
	}
	return splitBySizes(a, sizes, axis);
}
/**
* Split an array into multiple sub-arrays, allowing unevenly sized sections.
*
* If the axis does not divide evenly, the first sections are one element
* larger than the remaining sections.
*/
function arraySplit(a, indicesOrSections, axis = 0) {
	a = fudgeArray(a);
	axis = require_backend.checkAxis(axis, a.ndim);
	if (typeof indicesOrSections !== "number") return split$1(a, indicesOrSections, axis);
	const size = a.shape[axis];
	const partSize = Math.floor(size / indicesOrSections);
	const remainder = size % indicesOrSections;
	const sizes = [...require_backend.rep(remainder, partSize + 1), ...require_backend.rep(indicesOrSections - remainder, partSize)];
	return splitBySizes(a, sizes, axis);
}
/**
* Split an array into multiple sub-arrays along the depth (third) axis.
*
* Equivalent to `split()` with `axis=2`. The input array must have at least
* three dimensions.
*/
function dsplit(a, indicesOrSections) {
	a = fudgeArray(a);
	if (a.ndim < 3) throw new Error("dsplit only works on arrays of 3 or more dimensions");
	return split$1(a, indicesOrSections, 2);
}
/**
* Split an array into multiple sub-arrays horizontally (column-wise).
*
* Equivalent to `split()` along axis 1 for arrays of rank 2 or higher, or
* along axis 0 for rank-1 arrays.
*/
function hsplit(a, indicesOrSections) {
	a = fudgeArray(a);
	if (a.ndim === 0) throw new Error("hsplit only works on arrays of 1 or more dimensions");
	return split$1(a, indicesOrSections, a.ndim > 1 ? 1 : 0);
}
/**
* Split an array into multiple sub-arrays vertically (row-wise).
*
* Equivalent to `split` with `axis=0`.
*/
function vsplit(a, indicesOrSections) {
	return split$1(a, indicesOrSections, 0);
}
function splitBySizes(a, sizes, axis) {
	const results = [];
	for (let i = 0; i < sizes.length; i += 7) if (i === sizes.length) {
		results.push(a);
		break;
	} else if (i + 8 >= sizes.length) {
		results.push(...split$2(a, axis, sizes.slice(i)));
		break;
	} else {
		const groupSizes = [...sizes.slice(i, i + 7), sizes.slice(i + 7).reduce((x, y) => x + y, 0)];
		const outs = split$2(a, axis, groupSizes);
		results.push(...outs.slice(0, -1));
		a = outs[outs.length - 1];
	}
	return results;
}
/**
* Join a sequence of arrays along an existing axis.
*
* The arrays must have the same shape, except in the dimension corresponding to
* `axis` (the first, by default).
*
* No scalars can be passed to this function, as the axis is then ambiguous.
*/
function concatenate(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to concatenate");
	const shapes = xs.map(shape);
	axis = require_backend.checkAxis(axis, shapes[0].length);
	for (let i = 1; i < shapes.length; i++) if (shapes[i].length !== shapes[0].length || !shapes[i].every((d, j) => j === axis || d === shapes[0][j])) throw new Error(`Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`);
	let result = xs[0];
	for (let i = 1; i < xs.length; i += 7) {
		const group = xs.slice(i, i + 7);
		result = concatenate$1([result, ...group], axis);
	}
	return result;
}
/**
* Join a sequence of arrays along a new axis.
*
* The `axis` parameter specifies the index of the new axis in the dimensions of
* the result. For example, if `axis=0` it will be the first dimension and if
* `axis=-1` it will be the last dimension.
*
* All shapes must have the same shape.
*/
function stack(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to stack");
	const shapes = xs.map((x) => shape(x));
	if (!shapes.every((s) => require_backend.deepEqual(s, shapes[0]))) throw new Error(`Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`);
	axis = require_backend.checkAxis(axis, shapes[0].length + 1);
	const newShape = shapes[0].toSpliced(axis, 0, 1);
	return concatenate(xs.map((x) => fudgeArray(x).reshape(newShape)), axis);
}
/**
* Split an array into a sequence of arrays along the given axis.
*
* The `axis` parameter specifies the dimension of the input array to unstack
* along; it is removed from the shape of each output array. This is the
* inverse operation of `stack()`.
*/
function unstack(x, axis = 0) {
	x = fudgeArray(x);
	if (x.ndim === 0) throw new Error("unstack requires arrays with rank > 0");
	axis = require_backend.checkAxis(axis, x.ndim);
	const size = x.shape[axis];
	if (size === 0) {
		x.dispose();
		return [];
	}
	return split$1(x, size, axis).map((part) => squeeze(part, axis));
}
/**
* Horizontally stack arrays. Inputs are promoted to rank at least 1, then
* concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
*/
function hstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to hstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs);
	else if (nds[0] === 1) return concatenate(xs);
	else return concatenate(xs, 1);
}
/**
* Vertically stack arrays. Inputs are promoted to rank at least 2, then
* concatenated along axis 0.
*/
function vstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to vstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([-1, 1]);
	else if (nds[0] === 1) return stack(xs);
	else return concatenate(xs);
}
/**
* Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
* concatenated along axis 2.
*/
function dstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to dstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([
		1,
		1,
		-1
	]);
	else if (nds[0] === 1) {
		const ret = stack(xs, -1);
		return ret.reshape([1, ...ret.shape]);
	} else if (nds[0] === 2) return stack(xs, -1);
	else return concatenate(xs, 2);
}
/**
* Stack arrays column-wise. Inputs are promoted to rank at least 2, then
* concatenated along axis 1.
*/
function columnStack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to columnStack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([1, -1]);
	else if (nds[0] === 1) return stack(xs, -1);
	else return concatenate(xs, 1);
}
/** Flip an array vertically (axis=0). */
function flipud(x) {
	return flip(x, 0);
}
/** Flip an array horizontally (axis=1). */
function fliplr(x) {
	return flip(x, 1);
}
/** Rotate an array by 90 degrees in the plane specified by axes. */
function rot90(m, k = 1, axes = [0, 1]) {
	if (!Number.isInteger(k)) throw new Error(`rot90: k must be an integer`);
	if (!JsArray$2.isArray(axes) || axes.length !== 2) throw new Error(`rot90: axes must contain exactly two axes`);
	const a = fudgeArray(m);
	const [axis1, axis2] = require_backend.normalizeAxis(axes, a.ndim, false);
	switch ((k % 4 + 4) % 4) {
		case 0: return a;
		case 1: return swapaxes(flip(a, axis2), axis1, axis2);
		case 2: return flip(a, [axis1, axis2]);
		case 3: return flip(swapaxes(a, axis1, axis2), axis2);
		default: throw new Error(`rot90: unreachable ${k}`);
	}
}
/**
* Roll the elements of an array along a specified axis.
* @param a - Input array.
* @param shift - The number of places by which elements are shifted. If multiple,
* the shift is specified for each axis individually.
* @param axis - The axis or axes to roll. If `null`, the array is flattened,
* shifted, and then reshaped to its original shape.
*/
function roll(a, shift, axis = null) {
	a = fudgeArray(a);
	if (axis === null) return roll(a.ravel(), shift, 0).reshape(a.shape);
	axis = require_backend.normalizeAxis(axis, a.ndim, false);
	if (typeof shift === "number") shift = require_backend.rep(axis.length, shift);
	else if (shift.length !== axis.length) throw new Error(`roll: shift and axis must have the same length, got shift=${JSON.stringify(shift)} and axis=${JSON.stringify(axis)}`);
	for (const [s, ax] of require_backend.zip(shift, axis)) {
		if (!Number.isInteger(s)) throw new Error(`roll: shift must be an integer, got ${s}`);
		const n = a.shape[ax];
		const s1 = (s % n + n) % n;
		if (s1 !== 0) {
			const parts = split$1(a, [n - s1], ax);
			a = concatenate([parts[1], parts[0]], ax);
		}
	}
	return a;
}
/** Interchange two axes of an array. */
function swapaxes(a, axis1, axis2) {
	a = fudgeArray(a);
	axis1 = require_backend.checkAxis(axis1, a.ndim);
	axis2 = require_backend.checkAxis(axis2, a.ndim);
	if (axis1 === axis2) return a;
	const perm = require_backend.range(a.ndim);
	perm[axis1] = axis2;
	perm[axis2] = axis1;
	return transpose(a, perm);
}
/** Transpose the last two dimensions of an array. */
function matrixTranspose(a) {
	if (ndim(a) < 2) throw new Error(`matrixTranspose: input array must be at least 2D`);
	return moveaxis(a, -1, -2);
}
/** Return a 1-D flattened array containing the elements of the input. */
function ravel(a) {
	return fudgeArray(a).ravel();
}
/** Remove one or more length-1 axes from an array. */
function squeeze(a, axis = null) {
	const as = shape(a);
	if (axis === null) axis = require_backend.range(as.length).filter((i) => as[i] === 1);
	else if (typeof axis === "number") axis = [axis];
	axis = axis.map((a) => require_backend.checkAxis(a, as.length));
	for (const a of axis) if (as[a] !== 1) throw new Error("Cannot squeeze axis with size != 1");
	return reshape(a, as.filter((_, i) => !axis.includes(i)));
}
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
function expandDims(a, axis) {
	const as = shape(a);
	axis = typeof axis === "number" ? [axis] : axis;
	axis = require_backend.normalizeAxis(axis, as.length + axis.length);
	const newShape = [];
	let srcIdx = 0;
	for (let i = 0; i < as.length + axis.length; i++) if (axis.includes(i)) newShape.push(1);
	else newShape.push(as[srcIdx++]);
	return reshape(a, newShape);
}
/**
* Repeat each element of an array after themselves.
*
* If no axis is provided, use the flattened input array, and return a flat
* output array.
*/
function repeat(a, repeats, axis) {
	if (!Number.isInteger(repeats) || repeats < 0) throw new Error(`repeat: repeats must be a non-negative integer, got ${repeats}`);
	a = fudgeArray(a);
	if (axis === void 0) {
		a = ravel(a);
		axis = 0;
	}
	axis = require_backend.checkAxis(axis, a.ndim);
	if (repeats === 1) return a;
	const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
	const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
	return broadcast(a, broadcastedShape, [axis + 1]).reshape(finalShape);
}
/**
* Construct an array by repeating A the number of times given by reps.
*
* If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
* integers, the resulting array will have a shape of `(reps[0] * d1,
* reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
*/
function tile(a, reps) {
	a = fudgeArray(a);
	if (typeof reps === "number") reps = [reps];
	if (!reps.every((r) => Number.isInteger(r) && r >= 0)) throw new Error(`tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`);
	const ndiff = reps.length - a.ndim;
	if (ndiff > 0) a = a.reshape([...require_backend.rep(ndiff, 1), ...a.shape]);
	if (ndiff < 0) reps = [...require_backend.rep(-ndiff, 1), ...reps];
	const broadcastedShape = [];
	const broadcastAxes = [];
	for (let i = 0; i < a.ndim; i++) {
		if (reps[i] > 1) {
			broadcastedShape.push(reps[i]);
			broadcastAxes.push(broadcastedShape.length - 1);
		}
		broadcastedShape.push(a.shape[i]);
	}
	const finalShape = a.shape.map((d, i) => reps[i] * d);
	return broadcast(a, broadcastedShape, broadcastAxes).reshape(finalShape);
}
/**
* Broadcast an array to a shape, with NumPy-style broadcasing rules.
*
* In other words, this lets you append axes to the left, and/or expand
* dimensions where the shape is 1.
*/
function broadcastTo(a, shape) {
	const nd = ndim(a);
	if (shape.length < nd) throw new Error(`broadcastTo: target shape ${JSON.stringify(shape)} has fewer dimensions than input array: ${nd}`);
	return broadcast(a, shape, require_backend.range(shape.length - nd));
}
/** Broadcast input shapes to a common output shape. */
function broadcastShapes(...shapes) {
	if (shapes.length === 0) return [];
	return shapes.reduce(require_backend.generalBroadcast);
}
/** Broadcast arrays to a common shape. */
function broadcastArrays(...arrays) {
	const outShape = broadcastShapes(...arrays.map((a) => shape(a)));
	return arrays.map((a) => broadcastTo(a, outShape));
}
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
function diagonal(a, offset, axis1, axis2) {
	return fudgeArray(a).diagonal(offset, axis1, axis2);
}
/**
* Extract a diagonal or construct a diagonal array.
*
* If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
* array, return a 2D array with v on the k-th diagonal.
*/
function diag(v, k = 0) {
	const a = fudgeArray(v);
	if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
	if (a.ndim === 1) {
		const n = a.shape[0];
		const ret = where(eye(n).equal(1), a.ref, zerosLike(a));
		if (k > 0) return pad(ret, [[0, k], [k, 0]]);
		else if (k < 0) return pad(ret, [[-k, 0], [0, -k]]);
		else return ret;
	} else if (a.ndim === 2) return diagonal(a, k);
	else throw new Error("numpy.diag only supports 1D and 2D arrays");
}
/**
* Create a two-dimensional array with the flattened input on the k-th
* diagonal.
*/
function diagflat(v, k = 0) {
	return diag(ravel(v), k);
}
/**
* Return the indices to access the main diagonal of an array.
*
* This returns a list of `ndim` index arrays, each holding `[0, 1, ..., n-1]`.
* They can be used with advanced indexing to access the main diagonal of an
* array with `ndim` dimensions, each of length `n`.
*/
function diagIndices(n, ndim = 2) {
	if (!Number.isInteger(n) || n < 0) throw new Error(`n must be a nonnegative integer, got ${n}`);
	if (!Number.isInteger(ndim) || ndim < 0) throw new Error(`ndim must be a nonnegative integer, got ${ndim}`);
	if (ndim === 0) return [];
	const index = arange(n);
	return [index, ...require_backend.range(ndim - 1).map(() => index.ref)];
}
/**
* Return the indices to access the main diagonal of an array.
*
* The input array must be at least 2D, and all dimensions must be of equal
* length. Returns a list of `ndim` index arrays of dtype `int32`, which can be
* used to access the main diagonal of the array with `Array.slice()`.
*/
function diagIndicesFrom(arr) {
	const a = fudgeArray(arr);
	const nd = a.ndim;
	const aShape = a.shape;
	a.dispose();
	if (nd < 2) throw new Error(`diagIndicesFrom: input array must be at least 2D, got ${nd}D`);
	if (!aShape.every((s) => s === aShape[0])) throw new Error(`diagIndicesFrom: all dimensions of input must be equal, got shape ${JSON.stringify(aShape)}`);
	return diagIndices(aShape[0], nd);
}
/** Calculate the sum of the diagonal of an array along the given axes. */
function trace(a, offset = 0, axis1 = 0, axis2 = 1) {
	return diagonal(a, offset, axis1, axis2).sum(-1);
}
/**
* Return a sorted copy of an array.
*
* The array is sorted along a specified axis (the last by default). This may be
* an unstable sort, and it dispatches to device-specific implementation.
*/
function sort(a, axis = -1) {
	return fudgeArray(a).sort(axis);
}
/**
* Return indices that would sort an array. Unlike `sort`, this is guaranteed to
* be a stable sorting algorithm; it always returns the smaller index first in
* event of ties.
*
* Returns an array of `int32` indices.
*
* The array is sorted along a specified axis (the last by default).
*/
function argsort(a, axis = -1) {
	return fudgeArray(a).argsort(axis);
}
/**
* Append values to the end of an array.
*
* If `axis` is `null`, both inputs are flattened before concatenation.
*/
function append(arr, values, axis = null) {
	if (axis === null) return concatenate([ravel(arr), ravel(values)], 0);
	const a = fudgeArray(arr);
	axis = require_backend.checkAxis(axis, a.ndim);
	return concatenate([a, fudgeArray(values)], axis);
}
/**
* Take elements from an array along an axis.
*
* This is equivalent to advanced indexing with integer indices over that
* numbered axis. By default, the flattened array is used.
*/
function take(a, indices, axis = null) {
	if (axis === null) {
		a = ravel(a);
		axis = 0;
	}
	axis = require_backend.checkAxis(axis, ndim(a));
	return gather(a, [indices], [axis], axis);
}
/**
* Take values from an array by matching 1D index slices along an axis.
*
* The input array and indices must have the same rank. Dimensions other than
* `axis` follow standard broadcasting rules, and the output shape is the
* broadcasted shape with the size of `indices` along `axis`.
*/
function takeAlongAxis(a, indices, axis = -1) {
	a = fudgeArray(a);
	indices = fudgeArray(indices);
	if (a.ndim !== indices.ndim) {
		const aShape = a.shape;
		const indexShape = indices.shape;
		a.dispose();
		indices.dispose();
		throw new Error(`takeAlongAxis: input and indices must have the same rank, got ${JSON.stringify(aShape)} and ${JSON.stringify(indexShape)}`);
	}
	axis = require_backend.checkAxis(axis, a.ndim);
	const outShape = indices.shape.slice();
	for (let i = 0; i < a.ndim; i++) {
		if (i === axis) continue;
		const aDim = a.shape[i];
		const indexDim = indices.shape[i];
		if (aDim !== indexDim && aDim !== 1 && indexDim !== 1) {
			const aShape = a.shape;
			const indexShape = indices.shape;
			a.dispose();
			indices.dispose();
			throw new Error(`takeAlongAxis: non-axis dimensions must broadcast, got ${JSON.stringify(aShape)} and ${JSON.stringify(indexShape)}`);
		}
		outShape[i] = Math.max(aDim, indexDim);
	}
	const coords = [];
	for (let i = 0; i < a.ndim; i++) if (i === axis) coords.push(broadcastTo(indices, outShape));
	else {
		const shape = require_backend.rep(a.ndim, 1);
		shape[i] = a.shape[i];
		coords.push(broadcastTo(arange(0, a.shape[i], 1, {
			dtype: int32,
			device: a.device
		}).reshape(shape), outShape));
	}
	return gather(a, coords, require_backend.range(a.ndim), 0);
}
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
function select(condlist, choicelist, defaultValue = 0) {
	if (condlist.length !== choicelist.length) throw new Error(`select: condlist must have length equal to choicelist (${condlist.length} vs ${choicelist.length})`);
	if (condlist.length === 0) throw new Error("select: condlist must be non-empty");
	let output = fudgeArray(defaultValue);
	for (let i = condlist.length - 1; i >= 0; i--) output = where(astype(condlist[i], bool), choicelist[i], output);
	return output;
}
/**
* Apply a function to 1D array slices along an axis.
*
* This is implemented internally with applications of `jax.vmap()`, rather than
* iterating the function over a loop.
*/
function applyAlongAxis(func1d, axis, x) {
	const nd = ndim(x);
	axis = require_backend.checkAxis(axis, nd);
	let func = func1d;
	for (let i = 1; i < nd - axis; i++) func = vmap$1(func, i);
	for (let i = 0; i < axis; i++) func = vmap$1(func, 0);
	const y = func(fudgeArray(x));
	const post = nd - axis - 1;
	return y.transpose([
		...require_backend.range(axis),
		...require_backend.range(axis + post, y.ndim),
		...require_backend.range(axis, axis + post).reverse()
	]);
}
/** Apply a function repeatedly over specified axes. */
function applyOverAxes(func, a, axes) {
	a = fudgeArray(a);
	for (const axis of axes) {
		const b = func(a, axis);
		if (b.ndim === a.ndim) a = b;
		else if (b.ndim === a.ndim - 1) a = expandDims(b, axis);
		else throw new Error("function must return array of the correct shape");
	}
	return a;
}
/**
* Return if two arrays are element-wise equal within a tolerance.
*
* The formula used is `|actual - expected| <= atol + rtol * |expected|`, with
* NaN values comparing equal if `equalNaN` is true.
*/
function allclose(actual, expected, options) {
	const { rtol = 1e-5, atol = 1e-7, equalNaN = false } = options ?? {};
	const x = array(actual);
	const y = array(expected);
	if (!require_backend.deepEqual(x.shape, y.shape)) return false;
	const xData = x.dataSync();
	const yData = y.dataSync();
	for (let i = 0; i < xData.length; i++) {
		if (equalNaN ? isNaN(xData[i]) !== isNaN(yData[i]) : isNaN(xData[i]) || isNaN(yData[i])) return false;
		if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) return false;
	}
	return true;
}
/**
* Check if two arrays are element-wise equal.
*
* Returns False if the arrays have different shapes. If `equalNaN` is True,
* NaNs in the same position are considered equal.
*/
function arrayEqual(a1, a2, opts) {
	a1 = fudgeArray(a1);
	a2 = fudgeArray(a2);
	if (!require_backend.deepEqual(a1.shape, a2.shape)) {
		a1.dispose();
		a2.dispose();
		return array(false);
	}
	if (opts?.equalNaN) return where(isnan(a1.ref).mul(isnan(a2.ref)), true, equal(a1, a2)).all();
	return equal(a1, a2).all();
}
/**
* Check if two arrays are element-wise equal after broadcasting.
*
* Unlike `arrayEqual`, this allows inputs with different but
* broadcast-compatible shapes.
*/
function arrayEquiv(a1, a2) {
	a1 = fudgeArray(a1);
	a2 = fudgeArray(a2);
	try {
		const [b1, b2] = broadcastArrays(a1, a2);
		return equal(b1, b2).all();
	} catch {
		a1.dispose();
		a2.dispose();
		return array(false);
	}
}
/** Matrix product of two arrays. */
function matmul(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) throw new Error("matmul: x and y must be at least 1D");
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	const numBatchDims = Math.min(Math.max(x.ndim, 2), y.ndim) - 2;
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2],
		lhsBatchDims: require_backend.range(-2 - numBatchDims, -2),
		rhsBatchDims: require_backend.range(-2 - numBatchDims, -2)
	});
}
/** Matrix-vector product. x1 is [..., M, N], x2 is [..., N] → [..., M]. */
function matvec(x1, x2) {
	if (ndim(x1) < 2 || ndim(x2) < 1) throw new Error("matvec: x1 must be at least 2D and x2 at least 1D");
	return einsum("...mn,...n->...m", x1, x2);
}
/** Vector-matrix product. x1 is [..., N], x2 is [..., N, M] → [..., M]. */
function vecmat(x1, x2) {
	if (ndim(x1) < 1 || ndim(x2) < 2) throw new Error("vecmat: x1 must be at least 1D and x2 at least 2D");
	return einsum("...n,...nm->...m", x1, x2);
}
/** Dot product of two arrays. */
function dot$1(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) return multiply(x, y);
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2]
	});
}
/**
* Compute the tensor dot product of two N-dimensional arrays.
*
* The behavior is determined by `axes`. If an integer `k`, sum over the last
* `k` axes of x and the first `k` axes of y. If a tuple, then the first array
* corresponds to the axes of x and the second to the axes of y.
*/
function tensordot(x, y, axes = 2) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (typeof axes === "number") axes = [require_backend.range(-axes, 0), require_backend.range(axes)];
	return dot(x, y, {
		lhsContractingDims: axes[0],
		rhsContractingDims: axes[1]
	});
}
/**
* Einstein summation.
*
* This is a general API for performing tensor reductions, products,
* transpositions, and traces using Einstein notation for referring to named
* axes. See the docs for `numpy.einsum()` for more information.
*
* <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
*
* The full einsum API is implemented, including implicit and explicit output
* indices, ellipsis broadcasting, Unicode subscripts, and an optimal path
* ordering algorithm. It lowers to one or more calls to:
*
* - `jax.lax.dot()`
* - `jax.numpy.diagonal()`
* - `jax.numpy.sum()`
* - `jax.numpy.transpose()`
*/
function einsum(...args) {
	if (args.length === 0) throw new Error("einsum: must provide at least one argument");
	let input;
	let operands = [];
	if (typeof args[0] === "string") {
		operands = args.slice(1).map(fudgeArray);
		input = parseEinsumExpression(args[0], operands.map((x) => x.shape));
	} else {
		const n = args.length >> 1;
		const shapes = [];
		const lhsIndices = [];
		for (let i = 0; i < n; i++) {
			operands.push(fudgeArray(args[2 * i]));
			shapes.push(operands[i].shape);
			lhsIndices.push(args[2 * i + 1]);
		}
		let rhsIndex;
		if (args.length % 2 === 1) rhsIndex = args[2 * n];
		else {
			const indexCount = [];
			for (const i of lhsIndices.flat()) indexCount[i] = (indexCount[i] ?? 0) + 1;
			rhsIndex = [...indexCount.entries()].filter(([_, count]) => count === 1).map(([i, _]) => i);
		}
		input = {
			lhsIndices,
			rhsIndex,
			shapes
		};
	}
	const path = computeEinsumPath(input);
	if (require_backend.DEBUG >= 3) console.info(`einsum: computed path: ${path.approximateFlops} flops`);
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	const processSingleTensor = (ar, index, doNotReduce = []) => {
		index = index.slice();
		diag: while (true) {
			for (let i = 0; i < index.length; i++) {
				const idx = index[i];
				const j = index.indexOf(idx, i + 1);
				if (j !== -1) {
					ar = diagonal(ar, 0, i, j);
					index.splice(j, 1);
					index.splice(i, 1);
					index.push(idx);
					continue diag;
				}
			}
			break;
		}
		for (let i = index.length - 1; i >= 0; i--) {
			const idx = index[i];
			if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
				ar = sum(ar, i);
				index.splice(i, 1);
			}
		}
		return [ar, index];
	};
	for (const [i, j] of path.path) {
		let indexReduced = [];
		const indexGroup = [];
		for (const idx of [...indices[i], ...indices[j]]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		const [a, aidx] = processSingleTensor(operands[i], indices[i], indices[j]);
		const [b, bidx] = processSingleTensor(operands[j], indices[j], indices[i]);
		indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
		const indexBatch = aidx.filter((idx) => bidx.includes(idx) && !indexReduced.includes(idx));
		const result = dot(a, b, {
			lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
			rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
			lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
			rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx))
		});
		operands.push(result);
		indices.push([
			...indexBatch,
			...aidx.filter((idx) => !bidx.includes(idx)),
			...bidx.filter((idx) => !aidx.includes(idx))
		]);
		for (const idx of indices[indices.length - 1]) ++indexUsageCounts[idx];
	}
	for (const idx of indices[operands.length - 1]) --indexUsageCounts[idx];
	const [ar, index] = processSingleTensor(operands[operands.length - 1], indices[operands.length - 1]);
	const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
	return ar.transpose(finalPerm);
}
/**
* Compute the inner product of two arrays.
*
* Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
* contraction on the last axis.
*
* Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
*/
function inner(x, y) {
	return dot(fudgeArray(x), fudgeArray(y), {
		lhsContractingDims: [-1],
		rhsContractingDims: [-1]
	});
}
/**
* Compute the outer product of two arrays.
*
* If the input arrays are not 1D, they will be flattened. Returned array will
* be of shape `[x.size, y.size]`.
*/
function outer(x, y) {
	x = ravel(x);
	y = ravel(y);
	return multiply(x.reshape([x.shape[0], 1]), y);
}
/**
* Generate a Vandermonde matrix from a 1D input vector.
*
* @param x - Input vector.
* @param n - Number of columns. Defaults to `x.shape[0]`.
* @param increasing - If true, powers increase from left to right.
*/
function vander(x, { n, increasing = false } = {}) {
	x = fudgeArray(x);
	if (x.ndim !== 1) {
		const ndim = x.ndim;
		x.dispose();
		throw new Error(`vander: input must be 1D, got ${ndim}D`);
	}
	n ??= x.shape[0];
	if (!Number.isInteger(n) || n < 0) {
		x.dispose();
		throw new Error(`vander: n must be a non-negative integer, got ${n}`);
	}
	const rows = x.shape[0];
	if (n <= 1) return onesLike(x, { shape: [rows, n] });
	const result = cumulativeProd(broadcastTo(x.reshape([rows, 1]), [rows, n - 1]), {
		axis: 1,
		includeInitial: true
	});
	return increasing ? result : flip(result, 1);
}
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
function polyval(p, x) {
	p = fudgeArray(p);
	x = fudgeArray(x);
	if (p.ndim === 0) {
		const ndim = p.ndim;
		p.dispose();
		x.dispose();
		throw new Error(`polyval: coefficients must have at least one dimension, got ${ndim}D`);
	}
	const n = p.shape[0];
	const powers = vander(ravel(x), { n }).reshape([...x.shape, n]);
	return vecdot(moveaxis(p, 0, -1), powers);
}
/**
* Return the sum of two polynomials.
*
* The first axis contains polynomial coefficients, ordered from highest degree
* to the constant term. Remaining axes are broadcast batch dimensions. The
* shorter coefficient axis is aligned with the end of the longer one.
*/
function polyadd(a1, a2) {
	a1 = fudgeArray(a1);
	a2 = fudgeArray(a2);
	if (a1.ndim === 0 || a2.ndim === 0) {
		const [ndim1, ndim2] = [a1.ndim, a2.ndim];
		a1.dispose();
		a2.dispose();
		throw new Error(`polyadd: both inputs must be at least 1D arrays, got ${ndim1}D and ${ndim2}D`);
	}
	let base = a1;
	let update = a2;
	if (a2.shape[0] > a1.shape[0]) [base, update] = [a2, a1];
	const updateShape = [update.shape[0], ...base.shape.slice(1)];
	try {
		update = broadcastTo(update, updateShape);
	} catch (e) {
		base.dispose();
		update.dispose();
		throw e;
	}
	const leading = base.shape[0] - update.shape[0];
	if (leading > 0) update = pad(update, { 0: [leading, 0] });
	return add(base, update);
}
/**
* Return the difference of two polynomials.
*
* The first axis contains polynomial coefficients, ordered from highest degree
* to the constant term. Remaining axes are broadcast batch dimensions. The
* shorter coefficient axis is aligned with the end of the longer one.
*/
function polysub(a1, a2) {
	a1 = fudgeArray(a1);
	a2 = fudgeArray(a2);
	if (a1.ndim === 0 || a2.ndim === 0) {
		const [ndim1, ndim2] = [a1.ndim, a2.ndim];
		a1.dispose();
		a2.dispose();
		throw new Error(`polysub: both inputs must be at least 1D arrays, got ${ndim1}D and ${ndim2}D`);
	}
	const a2IsLonger = a2.shape[0] > a1.shape[0];
	let base = a1;
	let update = a2;
	if (a2IsLonger) [base, update] = [a2, a1];
	const updateShape = [update.shape[0], ...base.shape.slice(1)];
	try {
		update = broadcastTo(update, updateShape);
	} catch (e) {
		base.dispose();
		update.dispose();
		throw e;
	}
	const leading = base.shape[0] - update.shape[0];
	if (leading > 0) update = pad(update, { 0: [leading, 0] });
	return a2IsLonger ? subtract(update, base) : subtract(base, update);
}
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
function polyder(p, m = 1) {
	p = fudgeArray(p);
	if (!Number.isInteger(m) || m < 0) {
		p.dispose();
		throw new Error(`polyder: order of derivative must be a non-negative integer, got ${m}`);
	}
	if (!require_backend.isFloatDtype(p.dtype)) p = astype(p, float32);
	if (m === 0) return p;
	if (p.ndim === 0) {
		const ndim = p.ndim;
		p.dispose();
		throw new Error(`polyder: coefficients must have at least one dimension, got ${ndim}D`);
	}
	const n = p.shape[0];
	const length = Math.max(n - m, 0);
	const coeff = [];
	for (let j = 0; j < length; j++) {
		let c = 1;
		for (let i = 1; i <= m; i++) c *= n - j - i;
		coeff.push(c);
	}
	const batchBroadcastDims = require_backend.rep(p.ndim - 1, 1);
	const scale = array(coeff, {
		dtype: p.dtype,
		device: p.device
	}).reshape([length, ...batchBroadcastDims]);
	return multiply(p.slice([0, length]), scale);
}
/**
* Return the product of two polynomials.
*/
function polymul(a1, a2) {
	a1 = fudgeArray(a1);
	a2 = fudgeArray(a2);
	if (a1.ndim !== 1 || a2.ndim !== 1) {
		const [ndim1, ndim2] = [a1.ndim, a2.ndim];
		a1.dispose();
		a2.dispose();
		throw new Error(`polymul: both inputs must be 1D arrays, got ${ndim1}D and ${ndim2}D`);
	}
	const promotedDtype = resultType(a1, a2);
	const dtype = require_backend.isFloatDtype(promotedDtype) ? promotedDtype : "float32";
	a1 = a1.astype(dtype);
	a2 = a2.astype(dtype);
	if (a1.shape[0] === 0) {
		a1.dispose();
		a1 = zeros([1], { dtype });
	}
	if (a2.shape[0] === 0) {
		a2.dispose();
		a2 = zeros([1], { dtype });
	}
	return convolve(a1, a2, "full");
}
/**
* @function Compute the cross product of two arrays.
*
* Supports 2D (scalar result) and 3D cross products, with optional axis
* arguments. If `axis` is given, it overrides `axisa`, `axisb`, and `axisc`.
*/
const cross = jit$1(function cross(a, b, { axisa = -1, axisb = -1, axisc = -1, axis } = {}) {
	if (axis !== void 0) {
		axisa = axis;
		axisb = axis;
		axisc = axis;
	}
	axisa = require_backend.checkAxis(axisa, ndim(a));
	axisb = require_backend.checkAxis(axisb, ndim(b));
	a = moveaxis(a, axisa, -1);
	b = moveaxis(b, axisb, -1);
	const da = a.shape.at(-1);
	const db = b.shape.at(-1);
	if (da !== 2 && da !== 3 || db !== 2 && db !== 3) throw new Error(`cross: incompatible dimensions for cross product (got ${da} and ${db})`);
	if (da === 2 && db === 2) {
		const [a0, a1] = split$1(a, 2, -1);
		const [b0, b1] = split$1(b, 2, -1);
		return squeeze(a0.mul(b1).sub(a1.mul(b0)), -1);
	}
	if (da === 2) {
		const zeroShape = [...a.shape.slice(0, -1), 1];
		a = concatenate([a, zeros(zeroShape)], -1);
	}
	if (db === 2) {
		const zeroShape = [...b.shape.slice(0, -1), 1];
		b = concatenate([b, zeros(zeroShape)], -1);
	}
	const [a0, a1, a2] = split$1(a, 3, -1);
	const [b0, b1, b2] = split$1(b, 3, -1);
	return moveaxis(concatenate([
		a1.ref.mul(b2.ref).sub(a2.ref.mul(b1.ref)),
		a2.mul(b0.ref).sub(a0.ref.mul(b2)),
		a0.mul(b1).sub(a1.mul(b0))
	], -1), -1, axisc);
}, { staticArgnums: [2] });
/** Vector dot product of two arrays along a given axis. */
function vecdot(x, y, { axis } = {}) {
	const xaxis = require_backend.checkAxis(axis ?? -1, ndim(x));
	const yaxis = require_backend.checkAxis(axis ?? -1, ndim(y));
	if (shape(x)[xaxis] !== shape(y)[yaxis]) throw new Error(`vecdot: shapes ${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`);
	x = moveaxis(x, xaxis, -1);
	y = moveaxis(y, yaxis, -1);
	return dot$2(x, y);
}
/**
* Return the dot product of two vectors.
*
* Like vecdot() but flattens the arguments first into vectors.
*/
function vdot(x, y) {
	return dot$2(ravel(x), ravel(y));
}
function _convImpl(name, x, y, mode) {
	if (x.ndim !== 1 || y.ndim !== 1) throw new Error(`${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`);
	let flipOutput = false;
	if (x.shape[0] < y.shape[0]) {
		[x, y] = [y, x];
		if (name === "correlate") flipOutput = true;
	}
	if (name === "convolve") y = flip(y);
	let padding;
	if (mode === "valid") padding = "VALID";
	else if (mode === "same") padding = "SAME_LOWER";
	else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
	else throw new Error(`${name}: invalid mode ${mode}, expected "full", "same", or "valid"`);
	const z = conv(x.slice(null, null), y.slice(null, null), [1], padding).slice(0, 0);
	return flipOutput ? flip(z) : z;
}
/** Convolution of two one-dimensional arrays. */
function convolve(x, y, mode = "full") {
	return _convImpl("convolve", x, y, mode);
}
/** Correlation of two one dimensional arrays. */
function correlate(x, y, mode = "valid") {
	return _convImpl("correlate", x, y, mode);
}
/**
* Return a tuple of coordinate matrices from coordinate vectors.
*
* Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
* fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
*/
function meshgrid(xs, { indexing } = {}) {
	indexing ??= "xy";
	for (const x of xs) if (x.ndim !== 1) throw new Error(`meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`);
	if (xs.length <= 1) return xs;
	if (indexing === "xy") {
		const [a, b, ...rest] = xs;
		const [rb, ra, ...rrest] = meshgrid([
			b,
			a,
			...rest
		], { indexing: "ij" });
		return [
			ra,
			rb,
			...rrest
		];
	}
	const shape = xs.map((x) => x.shape[0]);
	return xs.map((x, i) => broadcast(x, shape, [...require_backend.range(i), ...require_backend.range(i + 1, xs.length)]));
}
function indices(dimensions, { dtype, device, sparse } = {}) {
	dtype = dtype ?? int32;
	if (dimensions.some((d) => !Number.isInteger(d) || d < 0)) throw new Error(`indices: dimensions must be non-negative integers, got ${JSON.stringify(dimensions)}`);
	const output = dimensions.map((dim, i) => {
		const shape = require_backend.rep(dimensions.length, 1);
		shape[i] = dim;
		const x = arange(0, dim, 1, {
			dtype,
			device
		}).reshape(shape);
		return sparse ? x : broadcastTo(x, dimensions);
	});
	if (sparse) return output;
	return output.length > 0 ? stack(output) : zeros([0], {
		dtype,
		device
	});
}
function fromfunction(func, shape, { dtype, device } = {}) {
	dtype = dtype ?? float32;
	if (shape.some((d) => !Number.isInteger(d) || d < 0)) throw new Error(`fromfunction: shape must be non-negative integers, got ${JSON.stringify(shape)}`);
	let f = func;
	for (let loopIndex = 0; loopIndex < shape.length; loopIndex++) {
		const mappedIndex = shape.length - 1 - loopIndex;
		const inputAxes = shape.map((_, dimensionIndex) => dimensionIndex === mappedIndex ? 0 : null);
		f = vmap$1(f, inputAxes);
	}
	return map(fudgeArray, f(...shape.map((s) => arange(0, s, 1, {
		dtype,
		device
	}))));
}
/**
* Clip (limit) the values in an array.
*
* Given an interval, values outside the interval are clipped to the interval
* edges. For example, if an interval of [0, 1] is specified, values smaller
* than 0 become 0, and values larger than 1 become 1.
*
* If either bound is undefined, it is ignored.
*/
function clip(a, min, max) {
	a = fudgeArray(a);
	if (max !== void 0) a = minimum(a, max);
	if (min !== void 0) a = maximum(a, min);
	return a;
}
/**
* Calculate the absolute value element-wise.
*
* This is the same function as `jax.numpy.abs()`.
*/
function absolute(x) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), x.ref.mul(-1), x);
}
/**
* Return an element-wise indication of sign of the input.
*
* NaN is returned for NaN input. Every comparison against a NaN is false, so it
* has to be selected explicitly rather than falling out of the sign tests.
*/
function sign(x) {
	x = fudgeArray(x);
	const signs = where(notEqual(x.ref, 0), where(less(x.ref, 0), -1, 1), 0);
	return where(isnan(x), NaN, signs);
}
/**
* Test element-wise whether the sign bit is set.
*
* Unlike comparisons with zero, this distinguishes `-0` from `0`. The sign of
* NaN is also preserved, except where the platform canonicalizes NaN: JS
* engines may do so when the cpu backend reads floats out of typed arrays, and
* conversions of f16 NaN on GPU backends may lose the sign as well.
*/
function signbit(x) {
	const arr = fudgeArray(x);
	switch (arr.dtype) {
		case "bool":
		case "uint32": return zerosLike(arr, { dtype: "bool" });
		case "int32": return less(arr, 0);
		default: return less(arr.astype("float32").view("int32"), 0);
	}
}
/**
* @function
* Return the value with the magnitude of x and the sign of y, element-wise.
*
* The sign is read from the sign bit of `y`, so `-0` counts as negative, and a
* zero or NaN `y` still keeps the magnitude of `x`.
*/
const copysign = jit$1(function copysign(x, y) {
	const magnitude = absolute(x);
	return where(signbit(y), magnitude.ref.mul(-1), magnitude);
});
/**
* @function
* Element-wise maximum of two arrays, ignoring NaNs.
*
* Unlike `maximum`, if one of the two elements is NaN, the other is returned.
* NaN is only propagated when both elements are NaN.
*/
const fmax = jit$1(function fmax(x1, x2) {
	return where(greater(x1.ref, x2.ref).add(isnan(x2.ref)), x1, x2);
});
/**
* @function
* Return element-wise minimum of the input arrays, ignoring NaNs.
*
* Unlike `minimum()`, if one of the elements is NaN, the other element is
* returned. NaN is only propagated when both elements are NaN.
*/
const fmin = jit$1(function fmin(x1, x2) {
	return where(logicalOr(less(x1.ref, x2.ref), isnan(x2.ref)), x1, x2);
});
/** @function Return element-wise positive values of the input (no-op). */
const positive = fudgeArray;
/**
* Return the Bartlett window of size M, a triangular taper.
*
* `w(n) = 1 - |2n - (M-1)| / (M-1)` for `0 <= n <= M-1`.
*/
function bartlett(M) {
	if (M < 0 || !Number.isInteger(M)) throw new RangeError(`Invalid window size for bartlett: ${M}. Must be a non-negative integer.`);
	if (M <= 1) return ones([M]);
	return subtract(1, absolute(linspace(-1, 1, M)));
}
/**
* Return the Hann window of size M, a taper with a weighted cosine bell.
*
* `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
*/
function hann(M) {
	return cos(linspace(0, 2 * Math.PI, M)).mul(-.5).add(.5);
}
/**
* Return the Blackman window of size M, a taper formed with the first three
* terms of a summation of cosines.
*
* `w(n) = 0.42 - 0.5 * cos(2πn/(M-1)) + 0.08 * cos(4πn/(M-1))` for
* `0 <= n <= M-1`.
*/
function blackman(M) {
	if (M < 0 || !Number.isInteger(M)) throw new RangeError(`Invalid window size for blackman: ${M}. Must be a non-negative integer.`);
	if (M <= 1) return ones([M]);
	return cos(linspace(0, 2 * Math.PI, M)).mul(-.5).add(cos(linspace(0, 4 * Math.PI, M)).mul(.08)).add(.42);
}
/**
* @function
* Compute the Heaviside step function. It is defined piecewise:
* - `heaviside(x1, x2) = 0` for `x1 < 0`,
* - `heaviside(x1, x2) = x2` for `x1 == 0`,
* - `heaviside(x1, x2) = 1` for `x1 > 0`.
*/
const heaviside = jit$1(function heaviside(x1, x2) {
	return where(less(x1.ref, 0), 0, where(equal(x1, 0), x2, 1));
});
/** Calculate element-wise square of the input array. */
function square(x) {
	x = fudgeArray(x);
	return x.ref.mul(x);
}
/** Element-wise tangent function (takes radians). */
function tan(x) {
	x = fudgeArray(x);
	return sin(x.ref).div(cos(x));
}
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
const sinc = jit$1(function sinc(x) {
	const pix = x.ref.mul(Math.PI);
	return where(equal(x, 0), 1, sin(pix.ref).div(pix));
});
/** Element-wise inverse cosine function (inverse of cos). */
function acos(x) {
	return subtract(pi / 2, asin(x));
}
/**
* @function
* Return element-wise hypotenuse for the given legs of a right triangle.
*
* In the original NumPy/JAX implementation, this function is more numerically
* stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
* stability improvements.
*/
const hypot = jit$1(function hypot(x1, x2) {
	return sqrt(square(x1).add(square(x2)));
});
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
const atan2 = jit$1(function atan2(y, x) {
	const r = sqrt(square(x.ref).add(square(y.ref)));
	const xNeg = less(x.ref, 0);
	const numer = where(xNeg.ref, r.ref.sub(x.ref), y.ref);
	const denom = where(xNeg, y, r.add(x));
	return atan(numer.div(denom)).mul(2);
});
/** Element-wise subtraction, with broadcasting. */
function subtract(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	return x.sub(y);
}
/** Calculates the floating-point division of x by y element-wise. */
function trueDivide(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) {
		x = x.astype("float32");
		y = y.astype("float32");
	}
	return x.div(y);
}
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
function floorDivide(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (require_backend.isFloatDtype(x.dtype) || require_backend.isFloatDtype(y.dtype)) return floor(trueDivide(x, y));
	return subtract(x, remainder(x.ref, y.ref)).div(y);
}
/**
* @function
* Calculate element-wise floating-point modulo operation.
*/
const fmod = jit$1(function fmod(x, y) {
	return x.ref.sub(y.ref.mul(idiv(x, y)));
});
/**
* @function
* Calculate element-wise remainder of the division (matches sign of y).
*/
const remainder = jit$1(function remainder(x, y) {
	return mod(mod(x, y.ref).add(y.ref), y);
});
/**
* Return element-wise quotient and remainder simultaneously.
*
* Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
*
* @param x - Dividend array.
* @param y - Divisor array.
* @returns Tuple of [quotient, remainder].
*/
function divmod(x, y) {
	const xArr = fudgeArray(x);
	const yArr = fudgeArray(y);
	return [floorDivide(xArr.ref, yArr.ref), remainder(xArr, yArr)];
}
/** Round input to the nearest integer towards zero. */
function trunc(x) {
	return idiv(x, 1);
}
/**
* Return the fractional and integral parts of an array, element-wise.
*
* Both parts have the same sign as the input, and satisfy
* `x == fractional + integral`. Integer inputs are promoted to float32.
*
* @param x - Input array.
* @returns Tuple of [fractional part, integral part].
*/
function modf(x) {
	x = fudgeArray(x);
	if (!require_backend.isFloatDtype(x.dtype)) x = x.astype("float32");
	const whole = trunc(x.ref);
	return [x.sub(whole.ref), whole];
}
/**
* @function
* Round to the given number of decimals.
*
* Uses banker's rounding (round half to even) to match NumPy/JAX behavior.
*/
const round = jit$1(function round(a, decimals = 0) {
	if (decimals === 0) return rint(a);
	const factor = 10 ** decimals;
	return rint(a.mul(factor)).mul(1 / factor);
}, { staticArgnums: [1] });
/**
* @function
* Round to the nearest integer, with ties going to the nearest even integer.
*/
const rint = jit$1(function rint(x) {
	const rounded = floor(x.ref.add(.5));
	const half = x.ref.sub(floor(x)).equal(.5);
	const odd = remainder(rounded.ref, 2).notEqual(0);
	return where(half.mul(odd), rounded.ref.sub(1), rounded);
});
/**
* Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
*
* This is the inverse of `frexp()`.
*/
function ldexp(x1, x2) {
	return multiply(x1, exp2(x2));
}
/**
* Decompose floating-point values into mantissa and two's exponent.
*
* The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
* `x != 0`, and the exponent is an integer such that
* `x = mantissa * 2**exponent`.
*/
function frexp(x) {
	x = fudgeArray(x);
	const absx = absolute(x.ref);
	const exponent = where(equal(x.ref, 0), 0, floor(log2(absx)).add(1).astype("int32"));
	return [x.div(exp2(exponent.ref.astype(x.dtype))), exponent];
}
/** Calculate `2**p` for all p in the input array. */
function exp2(p) {
	return exp(multiply(p, Math.LN2));
}
/** Return the base-2 logarithm of x, element-wise. */
function log2(x) {
	return log(x).mul(Math.LOG2E);
}
/** Return the base-10 logarithm of x, element-wise. */
function log10(x) {
	return log(x).mul(Math.LOG10E);
}
/** Calculate `exp(x) - 1` element-wise. */
function expm1(x) {
	x = fudgeArray(x);
	if (!require_backend.isFloatDtype(x.dtype)) x = x.astype("float32");
	const cutoff = x.dtype === "float64" ? .01 : .1;
	const useSeries = absolute(x.ref).lessEqual(cutoff);
	const seriesX = clip(x.ref, -cutoff, cutoff);
	return where(useSeries, seriesX.ref.mul(1 / 5040).add(1 / 720).mul(seriesX.ref).add(1 / 120).mul(seriesX.ref).add(1 / 24).mul(seriesX.ref).add(1 / 6).mul(seriesX.ref).add(1 / 2).mul(seriesX.ref).add(1).mul(seriesX), exp(x).sub(1));
}
/** Calculate the natural logarithm of `1 + x` element-wise. */
function log1p(x) {
	x = fudgeArray(x);
	if (!require_backend.isFloatDtype(x.dtype)) x = x.astype("float32");
	const cutoff = x.dtype === "float64" ? .01 : .2;
	const useSeries = absolute(x.ref).lessEqual(cutoff);
	const seriesX = clip(x.ref, -cutoff, cutoff);
	return where(useSeries, seriesX.ref.mul(-1 / 10).add(1 / 9).mul(seriesX.ref).sub(1 / 8).mul(seriesX.ref).add(1 / 7).mul(seriesX.ref).sub(1 / 6).mul(seriesX.ref).add(1 / 5).mul(seriesX.ref).sub(1 / 4).mul(seriesX.ref).add(1 / 3).mul(seriesX.ref).sub(1 / 2).mul(seriesX.ref).add(1).mul(seriesX), log(x.add(1)));
}
/**
* @function
* Compute `log(exp(x1) + exp(x2))` avoiding overflow.
*/
const logaddexp = jit$1(function logaddexp(x1, x2) {
	const xmax = maximum(x1.ref, x2.ref);
	return log(exp(x1.sub(xmax.ref)).add(exp(x2.sub(xmax.ref)))).add(xmax);
});
/**
* @function
* Logarithm of the sum of exponentials of inputs in base-2 avoiding overflow.
*/
const logaddexp2 = jit$1(function logaddexp2(x1, x2) {
	const xmax = maximum(x1.ref, x2.ref);
	return log2(exp2(x1.sub(xmax.ref)).add(exp2(x2.sub(xmax.ref)))).add(xmax);
});
/** Convert angles from degrees to radians. */
function deg2rad(x) {
	return multiply(x, pi / 180);
}
/** @function Alias of `jax.numpy.deg2rad()`. */
const radians = deg2rad;
/** Convert angles from radians to degrees. */
function rad2deg(x) {
	return multiply(x, 180 / pi);
}
/** @function Alias of `jax.numpy.rad2deg()`. */
const degrees = rad2deg;
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
function unwrap(p, discont = null, axis = -1, period = 2 * pi) {
	let x = fudgeArray(p);
	axis = require_backend.checkAxis(axis, x.ndim);
	if (!require_backend.isFloatDtype(x.dtype)) x = astype(x, float32);
	if (x.shape[axis] <= 1) return x;
	const toDtype = x.dtype === "float64" ? (v) => v : x.dtype === "float16" ? Math.f16round : Math.fround;
	const T = toDtype(period);
	const interval = toDtype(T / 2);
	const disc = toDtype(discont ?? T / 2);
	const skip = require_backend.rep(axis, []);
	const [head, tail] = split$1(x.ref, [1], axis);
	const dd = tail.ref.sub(x.slice(...skip, [0, -1]));
	const r = mod(dd.ref.add(interval), T);
	let ddmod = (T >= 0 ? where(less(r.ref, 0), r.ref.add(T), r) : where(greater(r.ref, 0), r.ref.add(T), r)).sub(interval);
	ddmod = where(logicalAnd(equal(ddmod.ref, -interval), greater(dd.ref, 0)), interval, ddmod);
	const phaseCorrection = where(less(absolute(dd.ref), disc), 0, ddmod.sub(dd));
	return concatenate([head, tail.add(cumsum(phaseCorrection, axis))], axis);
}
/**
* @function
* Computes first array raised to power of second array, element-wise.
*/
const power = jit$1(function power(x1, x2) {
	const x2i = trunc(x2.ref);
	const shouldBeNaN = multiply(x2.ref.notEqual(x2i.ref), x1.ref.less(0));
	const resultSign = where(mod(x2i, 2).notEqual(0), where(x1.ref.less(0), -1, 1), 1);
	return where(shouldBeNaN, nan, exp(log(absolute(x1)).mul(x2)).mul(resultSign));
});
/**
* Computes first array raised to power of second array, element-wise, after
* promoting inputs to a floating-point dtype.
*/
function floatPower(x1, x2) {
	x1 = fudgeArray(x1);
	x2 = fudgeArray(x2);
	const promotedDtype = promoteAvals(x1.aval, x2.aval).dtype;
	const dtype = require_backend.isFloatDtype(promotedDtype) ? promotedDtype : "float32";
	x1 = x1.astype(dtype);
	x2 = x2.astype(dtype);
	return power(x1, x2);
}
/** @function Calculate the element-wise cube root of the input array. */
const cbrt = jit$1(function cbrt(x) {
	const sgn = where(less(x.ref, 0), -1, 1);
	return sgn.ref.mul(exp(log(x.mul(sgn)).mul(1 / 3)));
});
/**
* @function
* Calculate element-wise hyperbolic sine of input.
*
* `sinh(x) = (exp(x) - exp(-x)) / 2`
*/
const sinh = jit$1(function sinh(x) {
	const ex = exp(x);
	const emx = reciprocal(ex.ref);
	return ex.sub(emx).mul(.5);
});
/**
* @function
* Calculate element-wise hyperbolic cosine of input.
*
* `cosh(x) = (exp(x) + exp(-x)) / 2`
*/
const cosh = jit$1(function cosh(x) {
	const ex = exp(x);
	const emx = reciprocal(ex.ref);
	return ex.add(emx).mul(.5);
});
/**
* @function
* Calculate element-wise hyperbolic tangent of input.
*
* `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
*/
const tanh = jit$1(function tanh(x) {
	const negsgn = where(less(x.ref, 0), 1, -1);
	const en2x = exp(x.mul(negsgn.ref).mul(2));
	return en2x.ref.sub(1).div(en2x.add(1)).mul(negsgn);
});
/**
* @function
* Calculate element-wise inverse hyperbolic sine of input.
*
* `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
*/
const arcsinh = jit$1(function arcsinh(x) {
	return log(x.ref.add(sqrt(square(x).add(1))));
});
/**
* @function
* Calculate element-wise inverse hyperbolic cosine of input.
*
* `arccosh(x) = ln(x + sqrt(x^2 - 1))`
*/
const arccosh = jit$1(function arccosh(x) {
	return log(x.ref.add(sqrt(square(x).sub(1))));
});
/**
* @function
* Calculate element-wise inverse hyperbolic tangent of input.
*
* `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
*/
const arctanh = jit$1(function arctanh(x) {
	return log(add(1, x.ref).div(subtract(1, x))).mul(.5);
});
/**
* Compute the variance of an array.
*
* The variance is computed for the flattened array by default, otherwise over
* the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function var_(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
	if (n === 0) throw new Error("var: cannot compute variance over zero-length axis");
	const mu = opts?.mean !== void 0 ? opts.mean : mean(x.ref, axis, { keepdims: true });
	return square(x.sub(mu)).sum(axis, { keepdims: opts?.keepdims }).mul(1 / (n - (opts?.correction ?? 0)));
}
/**
* Compute the variance of an array along the specified axis, ignoring NaNs.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of non-NaN elements. Slices where
* `N - correction` is not positive (including all-NaN slices) produce NaN.
*/
function nanvar(x, axis = null, opts) {
	x = fudgeArray(x);
	if (!require_backend.isFloatDtype(x.dtype)) x = x.astype(float32);
	const nanMask = isnan(x.ref);
	const count = sum(astype(logicalNot(nanMask.ref), int32), axis, { keepdims: opts?.keepdims });
	const mu = nanmean(x.ref, axis, { keepdims: true });
	const sq = sum(square(where(nanMask, 0, x.sub(mu))), axis, { keepdims: opts?.keepdims });
	const divisor = count.sub(opts?.correction ?? 0);
	const invalid = divisor.ref.lessEqual(0);
	const numerator = where(invalid.ref, NaN, sq);
	return numerator.div(astype(where(invalid, 1, divisor), numerator.dtype));
}
/**
* Compute the standard deviation of an array.
*
* The standard deviation is computed for the flattened array by default,
* otherwise over the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function std(x, axis = null, opts) {
	return sqrt(var_(x, axis, opts));
}
/**
* Compute the standard deviation of an array along the specified axis,
* ignoring NaNs.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of non-NaN elements. Slices where
* `N - correction` is not positive (including all-NaN slices) produce NaN.
*/
function nanstd(x, axis = null, opts) {
	return sqrt(nanvar(x, axis, opts));
}
/** Estimate the sample covariance of a set of variables. */
function cov(x, y = null, { rowvar = true } = {}) {
	x = fudgeArray(x);
	if (x.ndim === 1) x = x.reshape([1, x.shape[0]]);
	if (y !== null) {
		y = fudgeArray(y);
		if (y.ndim === 1) y = y.reshape([1, y.shape[0]]);
		x = vstack([x, y]);
	}
	if (!rowvar) x = x.transpose();
	const [_M, N] = x.shape;
	x = x.ref.sub(x.mean(1, { keepdims: true }));
	return dot$1(x.ref, x.transpose()).div(N - 1);
}
/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
function corrcoef(x, y) {
	const c = cov(x, y);
	const variances = diag(c.ref);
	const norm = sqrt(outer(variances.ref, variances));
	return c.div(norm);
}
/**
* Test whether each element of an array is also present in a second array.
*
* Returns a boolean array of the same shape as `element` that is true where
* the element is in `testElements`, and false otherwise. If `invert` is true,
* the result is negated, as if by `logicalNot(isin(...))`.
*/
function isin(element, testElements, opts) {
	element = fudgeArray(element);
	const outShape = element.shape;
	if (require_backend.prod(outShape) === 0) {
		fudgeArray(testElements).dispose();
		return astype(element, bool);
	}
	const result = any(equal(expandDims(ravel(element), 1), ravel(testElements)), 1);
	return reshape(opts?.invert ? logicalNot(result) : result, outShape);
}
/** Test element-wise for positive or negative infinity, return bool array. */
function isinf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.ref.equal(Infinity).add(x.equal(-Infinity)) : fullLike(x, false);
}
/** Test element-wise for NaN (Not a Number). */
function isnan(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.ref.notEqual(x) : fullLike(x, false);
}
/** Test element-wise for negative infinity, return bool array. */
function isneginf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike(x, false);
}
/** Test element-wise for positive infinity, return bool array. */
function isposinf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike(x, false);
}
/**
* Replace NaN and infinite entries in an array.
*
* By default, NaNs are replaced with `0.0`, and infinities are are substituted
* with the corresponding maximum or minimum finite values.
*/
function nanToNum(x, { nan = 0, posinf = null, neginf = null } = {}) {
	x = fudgeArray(x);
	x = where(isnan(x.ref), nan, x);
	posinf ??= require_backend.isFloatDtype(x.dtype) ? finfo(x.dtype).max : iinfo(x.dtype).max;
	neginf ??= require_backend.isFloatDtype(x.dtype) ? finfo(x.dtype).min : iinfo(x.dtype).min;
	x = where(isposinf(x.ref), posinf, x);
	x = where(isneginf(x.ref), neginf, x);
	return x;
}
/** Return the minimum of array elements along a given axis, ignoring NaNs. */
function nanmin(a, axis = null, opts) {
	a = fudgeArray(a);
	if (!require_backend.isFloatDtype(a.dtype)) return min(a, axis, opts);
	const nanMask = isnan(a.ref);
	const result = min(where(nanMask.ref, Infinity, a), axis, opts);
	return where(all(nanMask, axis, opts), NaN, result);
}
/**
* @function
* Test element-wise for finite values (not infinity or NaN).
*/
const isfinite = jit$1(function isfinite(x) {
	if (!require_backend.isFloatDtype(x.dtype)) return fullLike(x, true);
	return isnan(x.ref).add(isinf(x)).notEqual(true);
});
//#endregion
//#region src/library/lax-linalg.ts
var lax_linalg_exports = /* @__PURE__ */ __exportAll({
	cholesky: () => cholesky,
	eigh: () => eigh,
	lu: () => lu,
	svd: () => svd,
	triangularSolve: () => triangularSolve
});
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
function cholesky(a, { upper = false } = {}) {
	const L = cholesky$2(a);
	return upper ? moveaxis(L, -2, -1) : L;
}
/**
* Eigendecomposition of real symmetric matrices.
*
* This uses a cyclic Jacobi routine with an internal convergence check and a
* fixed maximum number of sweeps.
*
* Eigenvectors are returned as columns in the first result, and eigenvalues are
* returned in ascending order in the second result.
*/
function eigh(x, { lower = true, sortEigenvalues = true, symmetrizeInput = true } = {}) {
	x = fudgeArray(x);
	const n = require_backend.checkSquare("eigh", x.shape);
	if (!require_backend.isFloatDtype(x.dtype) || x.dtype === "float16") x = x.astype(float32);
	if (symmetrizeInput) x = x.ref.add(matrixTranspose(x)).mul(.5);
	else if (!lower) x = matrixTranspose(x);
	const batchShape = x.shape.slice(0, -2);
	let v;
	[x, v] = jacobiEigh(x, {
		maxSweeps: Math.max(8, 2 * n),
		tolerance: x.dtype === "float64" ? 1e-12 : 1e-6
	});
	const valuesUnsorted = diagonal(x, 0, -2, -1);
	if (!sortEigenvalues) return [v, valuesUnsorted];
	const order = argsort(valuesUnsorted.ref);
	const values = takeAlongAxis(valuesUnsorted, order.ref, -1);
	return [takeAlongAxis(v, order.reshape([
		...batchShape,
		1,
		n
	]), -1), values];
}
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
function svd(a, { computeUv = true, fullMatrices = false } = {}) {
	a = fudgeArray(a);
	if (a.ndim < 2) throw new Error(`svd: input must be at least 2D, got ${a}`);
	if (!require_backend.isFloatDtype(a.dtype) || a.dtype === "float16") a = a.astype(float32);
	const [m, n] = a.shape.slice(-2);
	if (fullMatrices && m !== n) throw new Error("svd: fullMatrices=true is only supported for square input");
	const batchShape = a.shape.slice(0, -2);
	const k = Math.min(m, n);
	const singularValues = (values) => sqrt(maximum(flip(values, -1), 0));
	const invSingularValues = (s) => where(greater(s.ref, 0), reciprocal(s.ref), 0);
	if (m >= n) {
		const [v, values] = eigh(matmul(matrixTranspose(a.ref), a.ref), { symmetrizeInput: false });
		const s = singularValues(values);
		if (!computeUv) {
			v.dispose();
			return s;
		}
		const vDesc = flip(v, -1);
		const invS = invSingularValues(s.ref);
		return [
			matmul(a, vDesc.ref).mul(invS.reshape([
				...batchShape,
				1,
				k
			])),
			s,
			matrixTranspose(vDesc)
		];
	} else {
		const [u, values] = eigh(matmul(a.ref, matrixTranspose(a.ref)), { symmetrizeInput: false });
		const s = singularValues(values);
		if (!computeUv) {
			u.dispose();
			return s;
		}
		const uDesc = flip(u, -1);
		const invS = invSingularValues(s.ref);
		return [
			uDesc,
			s,
			matmul(matrixTranspose(uDesc.ref), a).mul(invS.reshape([
				...batchShape,
				k,
				1
			]))
		];
	}
}
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
function lu(x) {
	return lu$1(x);
}
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
function triangularSolve(a, b, { leftSide = false, lower = false, transposeA = false, unitDiagonal = false } = {}) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	if (!leftSide) transposeA = !transposeA;
	else b = moveaxis(b, -2, -1);
	if (transposeA) {
		a = moveaxis(a, -2, -1);
		lower = !lower;
	}
	let x = triangularSolve$1(a, b, {
		lower,
		unitDiagonal
	});
	if (leftSide) x = moveaxis(x, -2, -1);
	return x;
}
//#endregion
//#region src/library/lax.ts
var lax_exports = /* @__PURE__ */ __exportAll({
	bitcastConvertType: () => bitcastConvertType,
	conv: () => conv,
	convGeneralDilated: () => convGeneralDilated,
	convTranspose: () => convTranspose,
	convWithGeneralPadding: () => convWithGeneralPadding,
	dot: () => dot,
	erf: () => erf,
	erfc: () => erfc,
	linalg: () => lax_linalg_exports,
	reduceWindow: () => reduceWindow,
	stopGradient: () => stopGradient,
	topK: () => topK
});
const JsArray$1 = globalThis.Array;
/** Elementwise bitcast an array into a new dtype. */
function bitcastConvertType(x, newDtype) {
	return fudgeArray(x).view(newDtype);
}
/**
* General dot product/contraction operator.
*
* Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
* `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
*/
function dot(lhs, rhs, { lhsContractingDims: lc = [], rhsContractingDims: rc = [], lhsBatchDims: lb = [], rhsBatchDims: rb = [] } = {}) {
	if (lc.length !== rc.length) throw new Error(`dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`);
	else if (lb.length !== rb.length) throw new Error(`dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`);
	lc = lc.map((a) => require_backend.checkAxis(a, lhs.ndim));
	rc = rc.map((a) => require_backend.checkAxis(a, rhs.ndim));
	lb = lb.map((a) => require_backend.checkAxis(a, lhs.ndim));
	rb = rb.map((a) => require_backend.checkAxis(a, rhs.ndim));
	if (lc.some((a) => lb.includes(a))) throw new Error(`dot: lhs contracting dims ${JSON.stringify(lc)} overlap with batch dims ${JSON.stringify(lb)}`);
	else if (rc.some((a) => rb.includes(a))) throw new Error(`dot: rhs contracting dims ${JSON.stringify(rc)} overlap with batch dims ${JSON.stringify(rb)}`);
	const lf = require_backend.range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
	const rf = require_backend.range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
	const lhs2 = lhs.transpose([
		...lb,
		...lf,
		...lc
	]);
	const rhs2 = rhs.transpose([
		...rb,
		...rf,
		...rc
	]);
	if (lc.length === 0) return mul(lhs2.reshape([
		...lb.map((a) => lhs.shape[a]),
		...lf.map((a) => lhs.shape[a]),
		...require_backend.rep(rf.length, 1)
	]), rhs2.reshape([
		...rb.map((a) => rhs.shape[a]),
		...require_backend.rep(lf.length, 1),
		...rf.map((a) => rhs.shape[a])
	]));
	const dotShapeX = lc.map((a) => lhs.shape[a]);
	const dotShapeY = rc.map((a) => rhs.shape[a]);
	if (!require_backend.deepEqual(dotShapeX, dotShapeY)) throw new Error(`dot: shapes not aligned along contracting dims: ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`);
	return dot$2(lhs2.reshape([
		...lb.map((a) => lhs.shape[a]),
		...lf.map((a) => lhs.shape[a]),
		...require_backend.rep(rf.length, 1),
		require_backend.prod(dotShapeX)
	]), rhs2.reshape([
		...rb.map((a) => rhs.shape[a]),
		...require_backend.rep(lf.length, 1),
		...rf.map((a) => rhs.shape[a]),
		require_backend.prod(dotShapeY)
	]));
}
function padtypeToPads(inShape, filterShape, strides, dilation, padding) {
	const padType = padding.toUpperCase();
	switch (padType) {
		case "VALID": return require_backend.rep(inShape.length, [0, 0]);
		case "SAME":
		case "SAME_LOWER": {
			const padSizes = require_backend.zipn(inShape.map((size, i) => Math.ceil(size / strides[i])), strides, filterShape, dilation, inShape).map(([o, s, k, d, i]) => Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i));
			if (padType === "SAME") return padSizes.map((size) => [size >> 1, size - (size >> 1)]);
			else return padSizes.map((size) => [size - (size >> 1), size >> 1]);
		}
		default: throw new Error(`Unknown padding type: ${padType}`);
	}
}
/**
* Lower a supported 1D ConvTranspose into a dense channel contraction and
* stride-phase overlap-add. Returns null to use the generic Conv otherwise.
*
* This is a specialized fast path for transposed convolutions.
*/
function lowerConvTranspose1d(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
	if (lhs.ndim !== 3 || rhs.ndim !== 3 || windowStrides.length !== 1 || windowStrides[0] !== 1 || lhsDilation.length !== 1 || lhsDilation[0] <= 1 || rhsDilation.length !== 1 || rhsDilation[0] !== 1 || padding.length !== 1 || padding[0][0] < 0 || padding[0][1] < 0 || lhs.shape.some((size) => size <= 0) || rhs.shape.some((size) => size <= 0)) return null;
	const outputShape = checkConvShape(lhs.shape, rhs.shape, {
		vmapDims: 0,
		strides: windowStrides,
		padding,
		lhsDilation,
		rhsDilation
	});
	if (outputShape.some((size) => size <= 0)) return null;
	const [batchSize, inputChannels, inputLength] = lhs.shape;
	const [outputChannels, , kernelSize] = rhs.shape;
	const stride = lhsDilation[0];
	const contributionCount = Math.ceil(kernelSize / stride);
	if (contributionCount > 8) return null;
	lhs = lhs.transpose([
		0,
		2,
		1
	]).reshape([
		batchSize,
		1,
		1,
		inputLength,
		inputChannels
	]);
	rhs = rhs.transpose([
		0,
		2,
		1
	]).reshape([
		1,
		outputChannels,
		kernelSize,
		1,
		inputChannels
	]);
	let columns = dot$2(lhs, rhs);
	columns = flip$1(columns.transpose([
		0,
		1,
		3,
		2
	]), [3]);
	columns = pad$1(columns, { 3: [0, contributionCount * stride - kernelSize] });
	columns = columns.reshape([
		batchSize,
		outputChannels,
		inputLength,
		contributionCount,
		stride
	]);
	let output = split$2(columns, 3, require_backend.rep(contributionCount, 1)).map((term, index) => {
		term = term.reshape([
			batchSize,
			outputChannels,
			inputLength,
			stride
		]);
		return pad$1(term, { 2: [index, contributionCount - 1 - index] });
	}).reduce((lhs, rhs) => lhs.add(rhs)).reshape([
		batchSize,
		outputChannels,
		-1
	]);
	const fullLength = (inputLength - 1) * stride + kernelSize;
	output = output.slice([], [], [0, fullLength]);
	const outputLength = outputShape[2];
	const offset = kernelSize - 1 - padding[0][0];
	const padLeft = Math.max(0, -offset);
	const padRight = Math.max(0, offset + outputLength - fullLength);
	output = pad$1(output, { 2: [padLeft, padRight] });
	return output.slice([], [], [offset + padLeft, offset + padLeft + outputLength]);
}
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
function convGeneralDilated(lhs, rhs, windowStrides, padding, { lhsDilation, rhsDilation, featureGroupCount = 1 } = {}) {
	if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
	if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
	if (typeof padding === "string") {
		if (lhsDilation?.some((d) => d !== 1)) throw new Error("String padding is not supported for transposed convolutions");
		padding = padtypeToPads(lhs.shape.slice(2), rhs.shape.slice(2), windowStrides, rhsDilation ?? require_backend.rep(rhs.ndim - 2, 1), padding);
	}
	if (featureGroupCount === 1) {
		const spatialDims = lhs.ndim - 2;
		const lowered = lowerConvTranspose1d(lhs, rhs, windowStrides, padding, lhsDilation ?? require_backend.rep(spatialDims, 1), rhsDilation ?? require_backend.rep(spatialDims, 1));
		if (lowered !== null) return lowered;
	}
	if (featureGroupCount !== 1) {
		const G = featureGroupCount;
		const [N, C_in, ...xs] = lhs.shape;
		const [C_out, C_in_per_group, ...ks] = rhs.shape;
		if (C_in % G !== 0) throw new Error(`featureGroupCount=${G} must divide input channels=${C_in}`);
		if (C_out % G !== 0) throw new Error(`featureGroupCount=${G} must divide output channels=${C_out}`);
		if (C_in / G !== C_in_per_group) throw new Error(`rhs input channels=${C_in_per_group} must equal lhs input channels / groups=${C_in / G}`);
		const result = conv$1(moveaxis$1(lhs.reshape([
			N,
			G,
			C_in / G,
			...xs
		]), 1, 0), rhs.reshape([
			G,
			C_out / G,
			C_in_per_group,
			...ks
		]), {
			vmapDims: 1,
			strides: windowStrides,
			padding,
			lhsDilation,
			rhsDilation
		});
		const ys = result.shape.slice(3);
		return moveaxis$1(result, 0, 1).reshape([
			N,
			C_out,
			...ys
		]);
	}
	return conv$1(lhs, rhs, {
		strides: windowStrides,
		padding,
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function convWithGeneralPadding(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding, {
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function conv(lhs, rhs, windowStrides, padding) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding);
}
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
function convTranspose(lhs, rhs, strides, padding, { rhsDilation, transposeKernel = false } = {}) {
	const kernelShape = rhs.shape.slice(2);
	rhsDilation = rhsDilation ?? require_backend.rep(kernelShape.length, 1);
	const pads = kernelShape.map((k, i) => Math.max(0, (k - 1) * rhsDilation[i] + 1)).map((k, i) => convTransposePadding(k, strides[i], typeof padding === "string" ? padding : padding[i]));
	if (transposeKernel) {
		rhs = flip$1(rhs, require_backend.range(2, rhs.ndim));
		rhs = moveaxis$1(rhs, 0, 1);
	}
	return convGeneralDilated(lhs, rhs, require_backend.rep(lhs.ndim - 2, 1), pads, {
		lhsDilation: strides,
		rhsDilation
	});
}
function convTransposePadding(k, s, padding) {
	let padLen;
	let pad1;
	if (padding === "SAME") {
		padLen = k + s - 2;
		pad1 = s > k - 1 ? k - 1 : Math.ceil(padLen / 2);
	} else if (padding === "VALID") {
		padLen = k + s - 2 + Math.max(k - s, 0);
		pad1 = k - 1;
	} else if (JsArray$1.isArray(padding)) {
		const pads = [k - 1 - padding[0], k - 1 - padding[1]];
		pad1 = pads[0];
		padLen = pads[0] + pads[1];
	} else throw new Error(`convTranspose: Invalid padding type ${padding}`);
	return [pad1, padLen - pad1];
}
/** Reduce a computation over padded windows. */
function reduceWindow(operand, computation, windowDimensions, windowStrides) {
	if (operand.ndim < windowDimensions.length) throw new Error(`Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`);
	if (!windowStrides) windowStrides = require_backend.rep(windowDimensions.length, 1);
	for (let i = 0; i < operand.ndim; i++) computation = vmap$1(computation, 0);
	return computation(bind1("pool", [operand], {
		window: windowDimensions,
		strides: windowStrides
	}));
}
/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
function erf(x) {
	return erf$1(x);
}
/**
* The complementary error function: `erfc(x) = 1 - erf(x)`.
*
* This function is more accurate than `1 - erf(x)` for large values of `x`,
* where `erf(x)` is very close to 1.
*/
function erfc(x) {
	return erfc$1(x);
}
/**
* Stops gradient computation.
*
* Behaves as the identity function but prevents the flow of gradients during
* forward or reverse-mode automatic differentiation.
*/
function stopGradient(x) {
	return stopGradient$1(x);
}
/**
* Returns top `k` values and their indices along the specified axis of operand.
*
* This is a _stable_ algorithm: If two elements are equal, the lower-index
* element appears first.
*
* @returns A tuple of `(values, indices)`, where `values` and `indices` have
* the same shape as `x`, except along `axis` where they have size `k`.
*/
function topK(x, k, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.checkAxis(axis, x.ndim);
	const size = x.shape[axis];
	if (k < 0 || k > size) throw new Error(`topK: k must be in the range [0, ${size}], got ${k}`);
	if (k === 0) {
		const outShape = x.shape.slice();
		outShape[axis] = 0;
		return [zerosLike$1(x.ref, { shape: outShape }), zerosLike$1(x, {
			dtype: "int32",
			shape: outShape
		})];
	}
	x = flip$1(x, [axis]);
	x = moveaxis$1(x, axis, -1);
	const [y, yi] = argsort$1(x);
	const extract = (a) => {
		a = a.slice(...require_backend.rep(a.ndim - 1, []), [-k]);
		return flip$1(moveaxis$1(a, -1, axis), [axis]);
	};
	return [extract(y), extract(yi.neg().add(size - 1))];
}
//#endregion
//#region src/library/nn.ts
var nn_exports = /* @__PURE__ */ __exportAll({
	celu: () => celu,
	dotProductAttention: () => dotProductAttention,
	elu: () => elu,
	gelu: () => gelu,
	glu: () => glu,
	hardSigmoid: () => hardSigmoid,
	hardSilu: () => hardSilu,
	hardSwish: () => hardSilu,
	hardTanh: () => hardTanh,
	identity: () => identity,
	leakyRelu: () => leakyRelu,
	log1mexp: () => log1mexp,
	logSigmoid: () => logSigmoid,
	logSoftmax: () => logSoftmax,
	logmeanexp: () => logmeanexp,
	logsumexp: () => logsumexp,
	mish: () => mish,
	oneHot: () => oneHot,
	relu: () => relu,
	relu6: () => relu6,
	selu: () => selu,
	sigmoid: () => sigmoid,
	silu: () => silu,
	softSign: () => softSign,
	softmax: () => softmax,
	softplus: () => softplus,
	sparsePlus: () => sparsePlus,
	sparseSigmoid: () => sparseSigmoid,
	squareplus: () => squareplus,
	standardize: () => standardize,
	swish: () => silu
});
/**
* Rectified Linear Unit (ReLU) activation function:
* `relu(x) = max(x, 0)`.
*/
function relu(x) {
	return maximum(x, 0);
}
/**
* Rectified Linear Unit 6 (ReLU6) activation function:
* `relu6(x) = min(max(x, 0), 6)`.
*/
function relu6(x) {
	return clip(x, 0, 6);
}
/**
* Sigmoid activation function, computed element-wise:
* `sigmoid(x) = 1 / (1 + exp(-x))`.
*
* Reference: https://en.wikipedia.org/wiki/Sigmoid_function
*/
function sigmoid(x) {
	return reciprocal(exp(negative(x)).add(1));
}
/**
* Softplus activation function:
* `softplus(x) = log(1 + exp(x))`.
*
* Reference: https://en.wikipedia.org/wiki/Softplus
*/
function softplus(x) {
	return log(exp(x).add(1));
}
/**
* @function
* Sparse plus function:
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x+1)**2 / 4`
* - When `x >= 1`: `x`
*/
const sparsePlus = jit$1((x) => {
	return where(x.ref.lessEqual(-1), 0, where(x.ref.less(1), square(x.ref.add(1)).mul(.25), x));
});
/**
* @function
* Sparse sigmoid activation function.
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x + 1) / 2`
* - When `x >= 1`: `1`
*/
const sparseSigmoid = jit$1((x) => {
	return clip(x.add(1).mul(.5), 0, 1);
});
/**
* Soft-sign activation function, computed element-wise:
* `softsign(x) = x / (|x| + 1)`.
*/
function softSign(x) {
	x = fudgeArray(x);
	return x.ref.div(absolute(x).add(1));
}
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
const silu = jit$1(function silu(x) {
	return x.ref.mul(sigmoid(x));
});
/**
* Log-sigmoid activation function, computed element-wise:
* `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
*/
function logSigmoid(x) {
	return negative(softplus(negative(x)));
}
/**
* @function
* Identity activation function. Returns the argument unmodified.
*/
const identity = fudgeArray;
/** Leaky rectified linear (ReLU) activation function */
function leakyRelu(x, negativeSlope = .01) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), x.ref.mul(negativeSlope), x);
}
/** Hard sigmoid activation function: `relu6(x+3)/6`. */
function hardSigmoid(x) {
	return relu6(add(x, 3)).mul(1 / 6);
}
/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
function hardSilu(x) {
	x = fudgeArray(x);
	return x.ref.mul(hardSigmoid(x));
}
/** Hard tanh activation function: `clip(x, -1, 1)`. */
function hardTanh(x) {
	return clip(x, -1, 1);
}
/**
* Exponential linear unit activation function.
*
* Computes the element-wise function:
* `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
*/
function elu(x, alpha = 1) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), exp(x.ref).sub(1).mul(alpha), x);
}
/**
* Continuously-differentiable exponential linear unit activation function.
*
* Computes the element-wise function:
* `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
*/
function celu(x, alpha = 1) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), exp(x.ref.div(alpha)).sub(1).mul(alpha), x);
}
/**
* @function
* Scaled exponential linear unit activation.
*
* Computes the element-wise function:
* `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
*
* Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
*/
const selu = jit$1(function selu(x) {
	return where(x.ref.less(0), expm1(x.ref).mul(1.6732632423543772), x).mul(1.0507009873554805);
});
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
const gelu = jit$1(function gelu(x, opts) {
	if (opts?.approximate ?? true) {
		const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
		return x.ref.mul(.5).mul(tanh(x.ref.mul(x.ref.mul(x).mul(.044715).add(1)).mul(SQRT_2_OVER_PI)).add(1));
	} else return x.ref.mul(.5).mul(erfc$1(negative(x.mul(Math.SQRT1_2))));
}, { staticArgnums: [1] });
/**
* Gated linear unit (GLU) activation function.
*
* Splits the `axis` dimension of the input into two halves, a and b, then
* computes `a * sigmoid(b)`.
*/
function glu(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.checkAxis(axis, x.ndim);
	const size = x.shape[axis];
	if (size % 2 !== 0) throw new Error(`glu: axis ${axis} of shape (${x.shape}) does not have even length`);
	const slice = x.shape.map((a) => [0, a]);
	const a = shrink(x.ref, slice.toSpliced(axis, 1, [0, size / 2]));
	const b = shrink(x, slice.toSpliced(axis, 1, [size / 2, size]));
	return a.mul(sigmoid(b));
}
/**
* Squareplus activation function.
*
* Computes the element-wise function:
* `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
*/
function squareplus(x, b = 4) {
	x = fudgeArray(x);
	return x.ref.add(sqrt(square(x).add(b))).mul(.5);
}
/**
* Mish activation function.
*
* Computes the element-wise function:
* `mish(x) = x * tanh(softplus(x))`
*/
function mish(x) {
	x = fudgeArray(x);
	return x.ref.mul(tanh(softplus(x)));
}
/**
* Softmax function. Computes the function which rescales elements to the range
* [0, 1] such that the elements along `axis` sum to 1.
*
* If `axis` is not specified, it defaults to the last axis.
*
* Reference: https://en.wikipedia.org/wiki/Softmax_function
*/
function softmax(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return onesLike(x);
	const xMax = max(x.ref, axis, { keepdims: true });
	const unnormalized = exp(x.sub(stopGradient$1(xMax)));
	return unnormalized.ref.div(unnormalized.sum(axis, { keepdims: true }));
}
/**
* Log-Softmax function.
*
* Computes the logarithm of the `softmax` function, which rescales elements to
* the range [-infinity, 0).
*
* If `axis` is not specified, it defaults to the last axis.
*/
function logSoftmax(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return zerosLike(x);
	const xMax = max(x.ref, axis, { keepdims: true });
	const shifted = x.sub(stopGradient$1(xMax));
	const shiftedLogsumexp = log(exp(shifted.ref).sum(axis, { keepdims: true }));
	return shifted.sub(shiftedLogsumexp);
}
/**
* Log-sum-exp reduction. Also a multivariate version of `softplus`.
*
* If no axis is specified, the reduction is performed over all elements. This
* convention differs from `jax.nn.logSoftmax()`.
*
* Reference: https://en.wikipedia.org/wiki/LogSumExp
*/
function logsumexp(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const xMax = stopGradient$1(max(x.ref, axis, { keepdims: true }));
	const shifted = x.sub(xMax.ref);
	const result = xMax.add(log(exp(shifted).sum(axis, { keepdims: true })));
	return opts?.keepdims ? result : squeeze(result, axis);
}
/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
function logmeanexp(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
	return logsumexp(x, axis, opts).sub(Math.log(n));
}
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
const log1mexp = jit$1(function log1mexp(x) {
	return where(x.ref.lessEqual(Math.LN2), log(negative(expm1(negative(x.ref)))), log1p(negative(exp(negative(x)))));
});
/**
* Standardizes input to zero mean and unit variance.
*
* By default, this is computed over the last axis. You can pass in a different
* axis, or `null` to standardize over all elements.
*
* Epsilon is added to denominator, it defaults to `1e-5` for stability.
*/
function standardize(x, axis = -1, opts = {}) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const mu = opts.mean !== void 0 ? fudgeArray(opts.mean) : x.ref.mean(axis, { keepdims: true });
	const sigma2 = opts.variance !== void 0 ? fudgeArray(opts.variance) : square(x.ref).mean(axis, { keepdims: true }).sub(square(mu.ref));
	return x.sub(mu).div(sqrt(sigma2.add(opts.epsilon ?? 1e-5)));
}
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
function oneHot(x, numClasses) {
	if (require_backend.isFloatDtype(x.dtype)) throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
	return eye(numClasses, void 0, { device: x.device }).slice(x);
}
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
function dotProductAttention(query, key, value, opts = {}) {
	query = fudgeArray(query);
	key = fudgeArray(key);
	value = fudgeArray(value);
	if (query.ndim !== 3 && query.ndim !== 4 || query.ndim !== key.ndim || query.ndim !== value.ndim) throw new Error(`dotProductAttention: expected all tensors to have rank 3 or 4, got Q=${query.aval}, K=${key.aval}, V=${value.aval}`);
	if (!require_backend.deepEqual(key.shape, value.shape)) throw new Error(`dotProductAttention: key and value shapes must match, got K=${key.shape}, V=${value.shape}`);
	const isRank3 = query.ndim === 3;
	if (isRank3) {
		query = expandDims(query, 0);
		key = expandDims(key, 0);
		value = expandDims(value, 0);
	}
	const [B, L, N, H] = query.shape;
	if (key.shape[0] !== B || key.shape[3] !== H) throw new Error(`dotProductAttention: query and key shapes mismatch, got Q=${query.aval}, K=${key.aval}`);
	const S = key.shape[1];
	const K = key.shape[2];
	if (N < K || N != K && N % K !== 0) throw new Error(`dotProductAttention: number of query heads N=${N} must be divisible by number of key/value heads K=${K} for GQA`);
	const G = N / K;
	key = repeat(key, G, 2);
	value = repeat(value, G, 2);
	const scale = opts.scale ?? 1 / Math.sqrt(H);
	let scores = einsum("BLNH,BSNH->BNLS", query, key).mul(scale);
	if (opts.bias !== void 0) scores = scores.add(opts.bias);
	if (opts.mask !== void 0) scores = where(opts.mask, scores, -Infinity);
	if (opts.isCausal) scores = where(tri(L, S, 0, { dtype: "bool" }), scores, -Infinity);
	if (opts.localWindowSize !== void 0) {
		const [before, after] = typeof opts.localWindowSize === "number" ? [opts.localWindowSize, opts.localWindowSize] : opts.localWindowSize;
		if (before < 0 || after < 0 || !Number.isInteger(before) || !Number.isInteger(after)) throw new Error(`dotProductAttention: localWindowSize values must be non-negative, got ${opts.localWindowSize}`);
		scores = where(tri(L, S, after, { dtype: "bool" }).mul(tri(L, S, -before - 1, { dtype: "bool" }).notEqual(true)), scores, -Infinity);
	}
	if (opts.querySeqLengths !== void 0) {
		const sl = expandDims(opts.querySeqLengths, [
			-1,
			-2,
			-3
		]);
		scores = where(arange(L).reshape([
			1,
			1,
			L,
			1
		]).less(sl), scores, -Infinity);
	}
	if (opts.keyValueSeqLengths !== void 0) {
		const sl = expandDims(opts.keyValueSeqLengths, [
			-1,
			-2,
			-3
		]);
		scores = where(arange(S).reshape([
			1,
			1,
			1,
			S
		]).less(sl), scores, -Infinity);
	}
	const out = einsum("BNLS,BSNH->BLNH", softmax(scores, -1), value);
	return isRank3 ? out.reshape([
		L,
		N,
		H
	]) : out;
}
//#endregion
//#region src/library/random.ts
var random_exports = /* @__PURE__ */ __exportAll({
	ball: () => ball,
	bernoulli: () => bernoulli,
	bits: () => bits,
	categorical: () => categorical,
	cauchy: () => cauchy,
	choice: () => choice,
	doubleSidedMaxwell: () => doubleSidedMaxwell,
	exponential: () => exponential,
	geometric: () => geometric,
	gumbel: () => gumbel,
	key: () => key,
	laplace: () => laplace,
	logistic: () => logistic,
	lognormal: () => lognormal,
	maxwell: () => maxwell,
	multivariateNormal: () => multivariateNormal,
	normal: () => normal,
	pareto: () => pareto,
	permutation: () => permutation,
	rademacher: () => rademacher,
	randint: () => randint,
	rayleigh: () => rayleigh,
	split: () => split,
	triangular: () => triangular,
	uniform: () => uniform,
	weibullMin: () => weibullMin
});
const JsArray = globalThis.Array;
function validateKeyShape(key, scalar = false) {
	if (key.ndim === 0) throw new Error("Key must have at least one dimension.");
	if (key.shape[key.shape.length - 1] !== 2) throw new Error(`Invalid key shape: ${key.shape}. Expected last dimension to be 2.`);
	if (scalar && key.shape.length > 1) throw new Error(`Expected a single PRNG key, but got a batch of keys with shape ${JSON.stringify(key.shape)} - use jax.vmap for batching.`);
	return key.shape.slice(0, -1);
}
function getK01(key) {
	const keyShape = validateKeyShape(key, true);
	let [k0, k1] = split$2(key, -1, [1, 1]);
	k0 = k0.reshape(keyShape);
	k1 = k1.reshape(keyShape);
	return [k0, k1];
}
/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
function key(seed) {
	seed = array(seed, { dtype: "uint32" });
	if (seed.ndim !== 0) throw new Error(`key: seed must be a scalar integer, but got shape ${seed.shape} - use jax.vmap for batching.`);
	const key = stack([0, seed]);
	if (key instanceof Array$1) key._realizeSource();
	return key;
}
/** Splits a PRNG key into `num` new keys by adding a leading axis. */
function split(key, num = 2) {
	const shape = typeof num === "number" ? [num] : num;
	for (const len of shape) if (len <= 0 || !Number.isInteger(len)) throw new Error(`Invalid split length: ${len}. Must be a positive integer.`);
	const [k0, k1] = getK01(key);
	return stack([randomBits(k0.ref, k1.ref, shape, 0), randomBits(k0, k1, shape, 1)], -1);
}
/** Sample uniform bits in the form of unsigned integers. */
function bits(key, shape = []) {
	const [k0, k1] = getK01(key);
	return randomBits(k0, k1, shape);
}
/**
* @function
* Sample uniform random values in [minval, maxval) with given shape.
*/
const uniform = jit$1(function uniform(key, shape = [], { minval = 0, maxval = 1 } = {}) {
	if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
	const rand = bitcast(bits(key, shape).div(array(512, {
		dtype: "uint32",
		device: key.device
	})).add(array(1065353216, {
		dtype: "uint32",
		device: key.device
	})), "float32").sub(1);
	if (minval === 0 && maxval === 1) return rand;
	else return rand.mul(maxval - minval).add(minval);
}, { staticArgnums: [1, 2] });
/**
* @function
* Sample points uniformly from the Euclidean unit ball in `d` dimensions.
*
* Only the Euclidean `p=2` case is currently supported.
*/
const ball = jit$1(function ball(key, d, { p = 2, shape = [] } = {}) {
	if (!Number.isInteger(d) || d <= 0) throw new Error(`ball: dimension must be a positive integer, got ${d}`);
	if (p !== 2) throw new Error("ball: only the Euclidean p=2 case is supported");
	const [k1, k2] = split(key, 2);
	const z = normal(k1, [...shape, d]);
	const norm = sqrt(z.ref.mul(z.ref).sum(-1, { keepdims: true }));
	const radius = exp(log(uniform(k2, [...shape, 1])).mul(1 / d));
	return z.div(norm).mul(radius);
}, { staticArgnums: [1, 2] });
/**
* Sample Bernoulli random variables with given mean (0,1 categorical).
*
* Returns a random Boolean array with the specified shape. `p` can be an array
* and must be broadcastable to `shape`.
*/
function bernoulli(key, p = .5, shape = []) {
	p = fudgeArray(p);
	return uniform(key, shape).less(p);
}
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
const categorical = jit$1(function categorical(key, logits, { axis = -1, shape, replace = true } = {}) {
	logits = fudgeArray(logits);
	axis = require_backend.checkAxis(axis, logits.ndim);
	const numCategories = logits.shape[axis];
	const batchShape = logits.shape.toSpliced(axis, 1);
	if (shape === void 0) shape = batchShape;
	else if (!require_backend.deepEqual(require_backend.generalBroadcast(shape, batchShape), shape)) throw new Error(`Shape ${shape} is not broadcast-compatible with batch shape ${batchShape}.`);
	const shapePrefix = shape.slice(0, shape.length - batchShape.length);
	if (replace) return argmax(gumbel(key, [...shapePrefix, ...logits.shape]).add(logits), axis + shapePrefix.length);
	else {
		const k = shapePrefix.reduce((a, b) => a * b, 1);
		if (k > numCategories) throw new Error(`Number of samples without replacement (${k}) cannot exceed number of categories (${numCategories}).`);
		const [values, indices] = topK(gumbel(key, logits.shape).add(logits), k, axis);
		values.dispose();
		return indices.reshape(shape);
	}
}, { staticArgnums: [2] });
/**
* @function
* Sample from a Cauchy distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = tan(π * (u - 0.5))` where u ~ Uniform(0, 1).
*/
const cauchy = jit$1(function cauchy(key, shape = []) {
	return tan(uniform(key, shape).sub(.5).mul(Math.PI));
}, { staticArgnums: [1] });
/**
* Sample from a population with optional replacement and optional probabilities.
*
* This implements the common JAX-compatible cases: integer populations and
* array populations along `axis`. Probabilities `p`, if provided, are sampled
* via `categorical(log(p))`.
*/
function choice(key, a, { shape = [], replace = true, p, axis = 0 } = {}) {
	let n;
	let values = null;
	if (typeof a === "number") {
		if (!Number.isInteger(a) || a < 0) throw new Error(`choice: a must be a non-negative integer, got ${a}`);
		n = a;
	} else {
		values = fudgeArray(a);
		axis = require_backend.checkAxis(axis, values.ndim);
		n = values.shape[axis];
	}
	let indices;
	if (p !== void 0) indices = categorical(key, log(p), {
		shape,
		replace
	});
	else if (replace) indices = randint(key, {
		minval: 0,
		maxval: n,
		shape
	});
	else {
		const k = shape.reduce((acc, x) => acc * x, 1);
		if (k > n) throw new Error(`Number of samples without replacement (${k}) cannot exceed population size (${n}).`);
		indices = permutation(key, n).slice([0, k]).reshape(shape);
	}
	if (values === null) return indices;
	const index = JsArray(axis).fill([]);
	index.push(indices);
	return values.slice(...index);
}
/**
* @function
* Sample double-sided Maxwell random values with the provided location and scale.
*/
const doubleSidedMaxwell = jit$1(function doubleSidedMaxwell(key, loc, scale, shape = []) {
	loc = fudgeArray(loc);
	scale = fudgeArray(scale);
	const [k1, k2] = split(key, 2);
	return rademacher(k1, {
		shape,
		dtype: "float32"
	}).mul(maxwell(k2, shape)).mul(scale).add(loc);
}, { staticArgnums: [3] });
/**
* @function
* Sample exponential random values according to `p(x) = exp(-x)`.
*/
const exponential = jit$1(function exponential(key, shape = []) {
	return negative(log1p(negative(uniform(key, shape))));
}, { staticArgnums: [1] });
/**
* @function
* Sample geometric random values: the number of trials until first success.
*/
const geometric = jit$1(function geometric(key, p, { shape = [], dtype = "int32" } = {}) {
	p = fudgeArray(p);
	return floor(log1p(negative(uniform(key, shape))).div(log1p(negative(p)))).add(1).astype(dtype);
}, { staticArgnums: [2] });
/**
* @function
* Sample from a Gumbel distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = -log(-log(u))` where u ~ Uniform(0, 1).
*/
const gumbel = jit$1(function gumbel(key, shape = []) {
	return negative(log(negative(log1p(negative(uniform(key, shape))))));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Laplace distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: the CDF is `F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))`.
* Inverting: `x = -sign(u - 0.5) * log(1 - 2 * |u - 0.5|)`.
*/
const laplace = jit$1(function laplace(key, shape = []) {
	const centered = uniform(key, shape).sub(.5);
	const s = sign(centered.ref);
	const absVal = absolute(centered);
	return s.mul(log1p(absVal.mul(-2)).mul(-1));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a logistic distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = log(u) - log(1-u)`.
*/
const logistic = jit$1(function logistic(key, shape = []) {
	const u = uniform(key, shape);
	return log(u.ref).sub(log1p(negative(u)));
}, { staticArgnums: [1] });
/**
* @function
* Sample log-normal random values: `exp(sigma * normal(key, shape))`.
*/
const lognormal = jit$1(function lognormal(key, sigma = 1, shape = []) {
	sigma = fudgeArray(sigma);
	return exp(normal(key, shape).mul(sigma));
}, { staticArgnums: [2] });
/**
* @function
* Sample Maxwell-distributed random values.
*/
const maxwell = jit$1(function maxwell(key, shape = []) {
	const z = normal(key, [...shape, 3]);
	return sqrt(z.ref.mul(z).sum(-1));
}, { staticArgnums: [1] });
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
const multivariateNormal = jit$1(function multivariateNormal(key, mean, cov, shape = []) {
	mean = fudgeArray(mean);
	cov = fudgeArray(cov);
	const n = mean.shape[mean.ndim - 1];
	if (cov.shape[cov.ndim - 1] !== n || cov.shape[cov.ndim - 2] !== n) throw new Error(`Invalid covariance shape: ${cov.shape}. Expected last two dimensions to be [${n}, ${n}].`);
	const outputShape = broadcastShapes(shape, mean.shape.slice(0, -1), cov.shape.slice(0, -2)).concat(n);
	return einsum("...ij,...j->...i", cholesky$1(cov), normal(key, outputShape)).add(mean);
}, { staticArgnums: [3] });
/**
* @function
* Sample random values according to `p(x) = 1/sqrt(2pi) * exp(-x^2/2)`.
*
* Unlike JAX, this uses the Box-Muller transform. JAX uses the erf_inv primitive instead and
* directly inverts the CDF, but we don't have support for that yet. Outputs will not be
* bitwise identical to JAX.
*/
const normal = jit$1(function normal(key, shape = []) {
	const [k1, k2] = split(key, 2);
	const u1 = uniform(k1, shape);
	const u2 = uniform(k2, shape);
	const radius = sqrt(log1p(negative(u1)).mul(-2));
	const theta = u2.mul(2 * Math.PI);
	return radius.mul(cos(theta));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Pareto distribution with shape parameter `b` and support [1, ∞).
*/
const pareto = jit$1(function pareto(key, b, shape = []) {
	b = fudgeArray(b);
	return exp(exponential(key, shape).div(b));
}, { staticArgnums: [2] });
/**
* Return a random permutation of an integer range or of an array along `axis`.
*/
function permutation(key, x, axis = 0) {
	if (typeof x === "number") {
		if (!Number.isInteger(x) || x < 0) throw new Error(`permutation: x must be a non-negative integer, got ${x}`);
		return argsort(uniform(key, [x])).astype("int32");
	}
	const arr = fudgeArray(x);
	axis = require_backend.checkAxis(axis, arr.ndim);
	const perm = permutation(key, arr.shape[axis]);
	const index = JsArray(axis).fill([]);
	index.push(perm);
	return arr.slice(...index);
}
/**
* @function
* Sample Rademacher random values, uniformly from {-1, 1}.
*/
const rademacher = jit$1(function rademacher(key, { shape = [], dtype = "int32" } = {}) {
	if (dtype === "uint32" || dtype === "bool") throw new Error(`rademacher: unsupported dtype ${dtype}`);
	const one = array(1, {
		dtype,
		device: key.device
	});
	const minusOne = array(-1, {
		dtype,
		device: key.device
	});
	return where(bernoulli(key, .5, shape), one, minusOne);
}, { staticArgnums: [1] });
/**
* @function
* Sample integer values uniformly from `[minval, maxval)`.
*
* This uses modulo reduction of uniform 32-bit random bits. For ranges that do
* not divide 2^32, this introduces a very small modulo bias.
*/
const randint = jit$1(function randint(key, { minval, maxval, shape = [], dtype = "int32" }) {
	if (!Number.isInteger(minval) || !Number.isInteger(maxval)) throw new Error("randint: minval and maxval must be integers");
	if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
	if (dtype !== "int32" && dtype !== "uint32") throw new Error(`randint: dtype must be int32 or uint32, got ${dtype}`);
	if (dtype === "uint32" && minval < 0) throw new Error("randint: uint32 dtype requires minval >= 0");
	const range = maxval - minval;
	return bits(key, shape).mod(range).astype(dtype).add(minval);
}, { staticArgnums: [1] });
/**
* @function
* Sample Rayleigh random values with the provided scale parameter.
*/
const rayleigh = jit$1(function rayleigh(key, scale = 1, shape = []) {
	scale = fudgeArray(scale);
	return sqrt(exponential(key, shape).mul(2)).mul(scale);
}, { staticArgnums: [2] });
/**
* @function
* Sample triangular random values on `[left, right]` with the given mode.
*/
const triangular = jit$1(function triangular(key, left, mode, right, shape = []) {
	left = fudgeArray(left);
	mode = fudgeArray(mode);
	right = fudgeArray(right);
	const u = uniform(key, shape);
	const width = right.ref.sub(left.ref);
	const leftSpan = mode.ref.sub(left.ref);
	const rightSpan = right.ref.sub(mode);
	const cutoff = leftSpan.ref.div(width.ref);
	return where(u.ref.less(cutoff), left.add(sqrt(u.ref.mul(width.ref).mul(leftSpan))), right.sub(sqrt(negative(u).add(1).mul(width).mul(rightSpan))));
}, { staticArgnums: [4] });
/**
* @function
* Sample Weibull minimum random values.
*
* Uses `scale * exponential(key) ** (1 / concentration)`.
*/
const weibullMin = jit$1(function weibullMin(key, scale, concentration, shape = []) {
	scale = fudgeArray(scale);
	concentration = fudgeArray(concentration);
	return scale.mul(exp(log(exponential(key, shape)).div(concentration)));
}, { staticArgnums: [3] });
//#endregion
//#region src/library/scipy-special.ts
var scipy_special_exports = /* @__PURE__ */ __exportAll({
	erf: () => erf,
	erfc: () => erfc,
	logSoftmax: () => logSoftmax,
	logit: () => logit,
	logsumexp: () => logsumexp,
	softmax: () => softmax
});
/**
* @function
* The logit function, `logit(p) = log(p / (1-p))`.
*/
const logit = jit$1(function logit(x) {
	return log(x.ref.div(subtract(1, x)));
});
//#endregion
//#region src/polyfills.ts
/** @file Polyfills for using this library. */
Symbol.dispose ??= Symbol.for("Symbol.dispose");
Symbol.asyncDispose ??= Symbol.for("Symbol.asyncDispose");
//#endregion
//#region src/index.ts
/** @namespace */
const profiler = {
	startTrace: require_backend.startTrace,
	stopTrace: require_backend.stopTrace
};
/**
* @function
* Compute the forward-mode Jacobian-vector product for a function.
*/
const jvp = jvp$1;
/**
* @function
* Vectorize an operation on a batched axis for one or more inputs.
*/
const vmap = vmap$1;
/**
* @function
* Compute the Jacobian evaluated column-by-column by forward-mode AD.
*/
const jacfwd = jacfwd$1;
/**
* @function
* Construct a Jaxpr by dynamically tracing a function with example inputs.
*/
const makeJaxpr = makeJaxpr$1;
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
const jit = jit$1;
/**
* @function
* Produce a local linear approximation to a function at a point using jvp() and
* partial evaluation.
*/
const linearize = linearize$1;
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
const vjp = vjp$1;
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
const grad = grad$1;
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
const valueAndGrad = valueAndGrad$1;
/**
* @function
* Compute the Jacobian evaluated row-by-row by reverse-mode AD.
*/
const jacrev = jacrev$1;
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
const hessian = hessian$1;
/**
* Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
*
* This can be used to wait for the results of an intermediate computation to
* finish. It's recommended to call this regularly in an iterative computation
* to avoid queueing up too many pending operations.
*
* Does not consume reference to the arrays.
*/
async function blockUntilReady(x) {
	const promises = [];
	for (const leaf of leaves(x)) if (leaf instanceof Array$1) promises.push(leaf.blockUntilReady());
	await Promise.all(promises);
	return x;
}
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
async function devicePut(x, device) {
	const [xflat, structure] = flatten(x);
	return unflatten(structure, await Promise.all(xflat.map((leaf) => {
		if (leaf instanceof Array$1) return device ? leaf._put(require_backend.getBackend(device)) : Promise.resolve(leaf);
		else return Promise.resolve(array(leaf, { device }));
	})));
}
//#endregion
exports.Array = Array$1;
exports.ClosedJaxpr = ClosedJaxpr;
exports.DType = require_backend.DType;
exports.Jaxpr = Jaxpr;
exports.blockUntilReady = blockUntilReady;
exports.defaultDevice = require_backend.defaultDevice;
exports.devicePut = devicePut;
exports.devices = require_backend.devices;
exports.getWebGPUDevice = require_backend.getWebGPUDevice;
exports.grad = grad;
exports.hessian = hessian;
exports.init = require_backend.init;
exports.jacfwd = jacfwd;
exports.jacobian = jacrev;
exports.jacrev = jacrev;
exports.jit = jit;
exports.jvp = jvp;
Object.defineProperty(exports, "lax", {
	enumerable: true,
	get: function() {
		return lax_exports;
	}
});
exports.linearize = linearize;
exports.makeJaxpr = makeJaxpr;
Object.defineProperty(exports, "nn", {
	enumerable: true,
	get: function() {
		return nn_exports;
	}
});
Object.defineProperty(exports, "numpy", {
	enumerable: true,
	get: function() {
		return numpy_exports;
	}
});
exports.profiler = profiler;
Object.defineProperty(exports, "random", {
	enumerable: true,
	get: function() {
		return random_exports;
	}
});
Object.defineProperty(exports, "scipySpecial", {
	enumerable: true,
	get: function() {
		return scipy_special_exports;
	}
});
exports.setDebug = require_backend.setDebug;
Object.defineProperty(exports, "tree", {
	enumerable: true,
	get: function() {
		return tree_exports;
	}
});
exports.valueAndGrad = valueAndGrad;
exports.vjp = vjp;
exports.vmap = vmap;
