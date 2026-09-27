const require_backend = require("./backend-TLHPgbCA.cjs");
//#region src/backend/webgpu/builtins.ts
const threefrySrc = `
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`;
const erfSrc = `
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`;
//#endregion
//#region src/backend/webgpu/codegen.ts
const headerWgsl = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();
/** Builder class for simple program generation with WGSL. */
var WgslBuilder = class {
	pushIndent = Symbol("pushIndent");
	popIndent = Symbol("popIndent");
	lines = [];
	#indent = "";
	emit(...lines) {
		for (const line of lines) if (line === this.pushIndent) this.#indent += "  ";
		else if (line === this.popIndent) this.#indent = this.#indent.slice(0, -2);
		else this.lines.push(line ? this.#indent + line : "");
	}
	emitPreamble(device, exps) {
		let hasFloat16 = false;
		let distinctOps = /* @__PURE__ */ new Map();
		for (const exp of exps) {
			if (exp == null) continue;
			hasFloat16 ||= exp.some((e) => e.dtype === "float16");
			distinctOps = require_backend.mapSetUnion(distinctOps, exp.distinctOps());
		}
		if (hasFloat16) {
			if (!device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
			this.emit("enable f16;");
		}
		this.emit(headerWgsl);
		if (distinctOps.has("Threefry2x32")) this.emit(threefrySrc);
		if (distinctOps.has("Erf") || distinctOps.has("Erfc")) this.emit(erfSrc);
		this.emit("");
	}
	/**
	* Insert phony assignments, in case some inputs are not in use.
	* <https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686>
	*/
	emitPhonyAssignments(args) {
		if (args.length > 0) this.emit(args.map((arg) => `_ = &${arg};`).join(" "));
	}
	toString() {
		return this.lines.join("\n");
	}
};
function dtypeToWgsl(dtype, storage = false) {
	switch (dtype) {
		case "bool": return storage ? "i32" : "bool";
		case "int32": return "i32";
		case "uint32": return "u32";
		case "float32": return "f32";
		case "float16": return "f16";
		default: throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
	}
}
function maxValueWgsl(dtype) {
	switch (dtype) {
		case "bool": return "1";
		case "int32": return "2147483647";
		case "uint32": return "4294967295u";
		case "float32": return "inf()";
		case "float16": return "f16(inf())";
		default: throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
	}
}
function constToWgsl(dtype, value) {
	if (dtype === "bool") return value ? "true" : "false";
	if (dtype === "int32") return value.toString();
	if (dtype === "uint32") return value.toString() + "u";
	if (dtype === "float32") {
		if (Number.isNaN(value)) return "nan()";
		if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
		return "f32(" + value.toString() + ")";
	}
	if (dtype === "float16") {
		if (Number.isNaN(value)) return "f16(nan())";
		if (!Number.isFinite(value)) return value > 0 ? "f16(inf())" : "f16(-inf())";
		return "f16(" + value.toString() + ")";
	}
	throw new Error(`Unsupported const dtype: ${dtype}`);
}
function reduceOpWgsl(op, dtype, a, b) {
	if (op === "Add") return `(${a} + ${b})`;
	if (op === "Mul") return `(${a} * ${b})`;
	if (op === "Min") return dtype === "bool" ? `(${a} && ${b})` : `min(${a}, ${b})`;
	if (op === "Max") return dtype === "bool" ? `(${a} || ${b})` : `max(${a}, ${b})`;
	throw new Error(`Unsupported reduction op: ${op}`);
}
/** Codegen for WebGPU expressions, linearizing AluOp into a kernel. */
var WgslExpCodegen = class {
	wb;
	args;
	#gensymCount = 0;
	#references = /* @__PURE__ */ new Map();
	#seen = /* @__PURE__ */ new Set();
	#context = /* @__PURE__ */ new Map();
	constructor(wb, args) {
		this.wb = wb;
		this.args = args;
	}
	#gensym() {
		return `alu${this.#gensymCount++}`;
	}
	#isGensym(text) {
		return text.match(/^alu[0-9]+$/);
	}
	/**
	* Count references for an expression.
	*
	* Used to get an ahead-of-time reference count for each node in the AluExp.
	* Expressions with reference count greater than 1 are stored in temporary
	* variables to avoid recomputation.
	*/
	countReferences(exp) {
		this.#references.set(exp, (this.#references.get(exp) ?? 0) + 1);
		if (!this.#seen.has(exp)) {
			this.#seen.add(exp);
			for (const src of exp.src) this.countReferences(src);
		}
	}
	reset() {
		this.#references.clear();
		this.#seen.clear();
		this.#context.clear();
	}
	/**
	* Generate code for an expression.
	*
	* Calls itself recursively and eliminates common subexpressions by storing
	* them in temporary variables, emitted to the current builder scope. This is
	* a side-effect that leads to multiline code generation.
	*/
	run(exp) {
		if (this.#context.has(exp)) return this.#context.get(exp);
		const { op, src, dtype, arg } = exp;
		let source = "";
		if (require_backend.AluGroup.Binary.has(op) || require_backend.AluGroup.Compare.has(op)) {
			const a = this.run(src[0]);
			const b = this.run(src[1]);
			if (op === "Add") if (dtype === "bool") source = `(${a} || ${b})`;
			else source = `(${a} + ${b})`;
			else if (op === "Sub") source = `(${a} - ${b})`;
			else if (op === "Mul") if (dtype === "bool") source = `(${a} && ${b})`;
			else source = `(${a} * ${b})`;
			else if (op === "Idiv") source = require_backend.isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
			else if (op === "Mod") source = `(${a} % ${b})`;
			else if (op === "Min") if (dtype === "bool") source = `(${a} && ${b})`;
			else source = `min(${require_backend.strip1(a)}, ${require_backend.strip1(b)})`;
			else if (op === "Max") if (dtype === "bool") source = `(${a} || ${b})`;
			else source = `max(${require_backend.strip1(a)}, ${require_backend.strip1(b)})`;
			else if (op === "BitCombine") if (arg === "and") source = `(${a} & ${b})`;
			else if (arg === "or") source = `(${a} | ${b})`;
			else source = dtype === "bool" ? `(${a} != ${b})` : `(${a} ^ ${b})`;
			else if (op === "BitShift") if (arg === "shl") source = `(${a} << ${b})`;
			else source = `(${a} >> ${b})`;
			else if (op === "Cmplt") source = `(${a} < ${b})`;
			else if (op === "Cmpne") if (require_backend.isFloatDtype(src[0].dtype)) {
				const x = this.#isGensym(a) ? a : this.#gensym();
				if (x !== a) this.wb.emit(`let ${x} = ${a};`);
				source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
			} else source = `(${a} != ${b})`;
		} else if (require_backend.AluGroup.Unary.has(op)) if (op === "Reciprocal" && src[0].op === "Sqrt") source = `inverseSqrt(${this.run(src[0].src[0])})`;
		else {
			const a = this.run(src[0]);
			if (op === "BitCount") source = `i32(countOneBits(${require_backend.strip1(a)}))`;
			else if (op === "Sin") source = `sin(${require_backend.strip1(a)})`;
			else if (op === "Cos") source = `cos(${require_backend.strip1(a)})`;
			else if (op === "Asin") source = `asin(${require_backend.strip1(a)})`;
			else if (op === "Atan") source = `atan(${require_backend.strip1(a)})`;
			else if (op === "Exp") source = `exp(${require_backend.strip1(a)})`;
			else if (op === "Log") source = `log(${require_backend.strip1(a)})`;
			else if (op === "Erf" || op === "Erfc") {
				const funcName = op === "Erf" ? "erf" : "erfc";
				if (dtype !== "float32") source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${require_backend.strip1(a)})))`;
				else source = `${funcName}(${require_backend.strip1(a)})`;
			} else if (op === "Sqrt") source = `sqrt(${require_backend.strip1(a)})`;
			else if (op === "Reciprocal") source = `(1.0 / ${a})`;
			else if (op === "Floor") source = `floor(${require_backend.strip1(a)})`;
			else if (op === "Ceil") source = `ceil(${require_backend.strip1(a)})`;
			else if (op === "Cast") {
				const srcTy = dtypeToWgsl(src[0].dtype);
				const dstTy = dtypeToWgsl(dtype);
				if (require_backend.isFloatDtype(src[0].dtype) && !(require_backend.isFloatDtype(dtype) || dtype === "bool")) {
					const maxVal = maxValueWgsl(dtype);
					const x = this.#isGensym(a) ? a : this.#gensym();
					if (x !== a) this.wb.emit(`let ${x}: ${srcTy} = ${require_backend.strip1(a)};`);
					source = `select(${dstTy}(${x}), ${maxVal}, ${x} >= ${srcTy}(${maxVal}))`;
				} else source = `${dstTy}(${require_backend.strip1(a)})`;
			} else if (op === "Bitcast") source = `bitcast<${dtypeToWgsl(dtype)}>(${require_backend.strip1(a)})`;
		}
		else if (op === "Where") source = `select(${require_backend.strip1(this.run(src[2]))}, ${require_backend.strip1(this.run(src[1]))}, ${require_backend.strip1(this.run(src[0]))})`;
		else if (op === "Threefry2x32") {
			const x = this.#gensym();
			const [k0, k1, c0, c1] = src.map((x) => require_backend.strip1(this.run(x)));
			this.wb.emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
			if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
			else if (arg === 0) source = `${x}.x`;
			else if (arg === 1) source = `${x}.y`;
			else throw new require_backend.UnsupportedOpError(op, dtype, "webgpu", arg);
		} else if (op === "Const") return constToWgsl(dtype, arg);
		else if (op === "Special") return arg[0];
		else if (op === "Variable") return arg;
		else if (op === "GlobalIndex") {
			source = `${this.args[arg[0]]}[${require_backend.strip1(this.run(src[0]))}]`;
			if (dtype === "bool") source = `(${source} != 0)`;
		}
		if (!source) throw new require_backend.UnsupportedOpError(op, dtype, "webgpu", arg);
		const typeName = dtypeToWgsl(dtype);
		if ((this.#references.get(exp) ?? 0) > 1) {
			const name = this.#gensym();
			this.#context.set(exp, name);
			this.wb.emit(`let ${name}: ${typeName} = ${require_backend.strip1(source)};`);
			return name;
		} else {
			this.#context.set(exp, source);
			return source;
		}
	}
};
const gridOffsetY = 16384;
function calculateGrid(gridSize) {
	let gridX = gridSize;
	let gridY = 1;
	if (gridSize > 65535) {
		gridX = gridOffsetY;
		gridY = Math.ceil(gridSize / gridOffsetY);
	}
	return [gridX, gridY];
}
//#endregion
//#region src/backend/webgpu/firefoxPoll.ts
const NUDGE_INTERVAL_MS = 1;
const BUSY_MS = 20;
const BACKOFF_MS = 5;
const DEADLINE_MS = 1e3;
const isFirefox = typeof navigator !== "undefined" && navigator.userAgent.includes("Firefox") && !navigator.userAgent.includes("Seamonkey");
let channel;
const resolvers = [];
function yieldTask() {
	const port = (channel ??= (() => {
		const created = new MessageChannel();
		created.port1.onmessage = () => resolvers.shift()?.();
		created.port1.start();
		return created;
	})()).port2;
	return new Promise((resolve) => {
		resolvers.push(resolve);
		port.postMessage(0);
	});
}
/** Map a staging buffer for reading, without Firefox's ~100 ms poll latency. */
async function mapAsyncRead(device, staging) {
	const mapped = staging.mapAsync(GPUMapMode.READ);
	if (!isFirefox) return mapped;
	let settled = false;
	const tracked = mapped.finally(() => {
		settled = true;
	});
	await yieldTask();
	const start = performance.now();
	let lastNudge = -Infinity;
	while (!settled) {
		const elapsed = performance.now() - start;
		if (elapsed >= DEADLINE_MS) break;
		if (performance.now() - lastNudge >= NUDGE_INTERVAL_MS) {
			lastNudge = performance.now();
			device.queue.submit([device.createCommandEncoder().finish()]);
		}
		if (elapsed < BUSY_MS) await yieldTask();
		else await new Promise((resolve) => setTimeout(resolve, BACKOFF_MS));
	}
	return tracked;
}
//#endregion
//#region src/backend/webgpu/nullaryKernel.ts
function uniformDtype(dtype) {
	if (dtype === "float16") return "float32";
	if (dtype === "bool") return "int32";
	return dtype;
}
function replacementFor({ name, dtype, uniformDtype }) {
	const value = require_backend.AluExp.variable(uniformDtype, `uniforms.${name}`);
	if (dtype === "float16") return require_backend.AluExp.cast("float16", value);
	if (dtype === "bool") return require_backend.AluExp.cmpne(value, require_backend.AluExp.i32(0));
	return value;
}
function writeUniform(view, offset, dtype, value) {
	switch (dtype) {
		case "float32":
			view.setFloat32(offset, value, true);
			break;
		case "int32":
			view.setInt32(offset, value, true);
			break;
		case "uint32":
			view.setUint32(offset, value, true);
			break;
		default: throw new Error(`Unsupported dtype for constant uniform: ${dtype}`);
	}
}
function liftConstants(exp) {
	const uniforms = [];
	return [exp.rewrite((node) => {
		if (node.op !== "Const" || node.arg === 0) return;
		const uniform = {
			name: `c${uniforms.length}`,
			dtype: node.dtype,
			uniformDtype: uniformDtype(node.dtype),
			value: node.arg
		};
		uniforms.push(uniform);
		return replacementFor(uniform);
	}), uniforms];
}
function uniformsData(uniforms) {
	const data = new Uint8Array(uniforms.length * 4);
	const view = new DataView(data.buffer);
	uniforms.forEach((u, i) => writeUniform(view, i * 4, u.uniformDtype, u.value));
	return data;
}
function nullaryKernelSource(device, kernel) {
	if (kernel.nargs !== 0 || kernel.reduction) return null;
	let exp = kernel.exp.substitute({ gidx: require_backend.AluExp.special("int32", "gidx", kernel.size) }).simplify();
	let uniforms = [];
	[exp, uniforms] = liftConstants(exp);
	const wb = new WgslBuilder();
	wb.emitPreamble(device, [exp]);
	if (uniforms.length > 0) wb.emit("struct Uniforms {", wb.pushIndent, ...uniforms.map((u) => `${u.name}: ${dtypeToWgsl(u.uniformDtype)},`), wb.popIndent, "}\n");
	const resultTy = dtypeToWgsl(kernel.dtype, true);
	wb.emit(`@group(0) @binding(0) var<storage, read_write> result : array<${resultTy}>;`);
	if (uniforms.length > 0) wb.emit(`@group(1) @binding(0) var<uniform> uniforms: Uniforms;`);
	const workgroupSize = require_backend.findPow2(kernel.size, 256);
	const [gridX, gridY] = calculateGrid(Math.ceil(kernel.size / workgroupSize));
	wb.emit("", `@compute @workgroup_size(${workgroupSize})`, "fn main(@builtin(global_invocation_id) id : vec3<u32>) {", wb.pushIndent);
	if (gridY === 1) wb.emit(`if (id.x >= ${kernel.size}) { return; }`, "let gidx: i32 = i32(id.x);");
	else {
		const sizeX = gridX * workgroupSize;
		wb.emit(`if (${sizeX} * id.y + id.x >= ${kernel.size}) { return; }`, `let gidx: i32 = i32(${sizeX} * id.y + id.x);`);
	}
	const gen = new WgslExpCodegen(wb, []);
	gen.countReferences(exp);
	let rhs = require_backend.strip1(gen.run(exp));
	if (resultTy !== dtypeToWgsl(exp.dtype)) rhs = `${resultTy}(${rhs})`;
	wb.emit(`result[gidx] = ${rhs};`, wb.popIndent, "}");
	return {
		code: wb.toString(),
		numInputs: 0,
		numOutputs: 1,
		hasUniform: uniforms.length > 0,
		passes: [{
			grid: [gridX, gridY],
			uniform: uniforms.length > 0 ? uniformsData(uniforms) : void 0
		}]
	};
}
//#endregion
//#region src/backend/webgpu/reader.ts
/**
* Graphics state used to synchronously read data from WebGPU buffers.
*
* This trick is borrowed from TensorFlow.js. Basically, the idea is to create
* an offscreen canvas with one pixel for every 4 bytes ("device storage"), then
* configure it with a WebGPU context. Copy the buffer to a texture, then draw
* the canvas onto another offscreen canvas with '2d' context ("host storage").
*
* Once it's on host storage, we can use `getImageData()` to read the pixels
* from the image directly.
*
* We use 256x256 canvases here (256 KiB). The performance of this is bad
* because it involves multiple data copies, but it still works. We also
* actually need to copy the image twice: once in "opaque" mode for the RGB
* values, and once in "premultiplied" mode for the alpha channel.
*
* https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L379
*/
var SyncReader = class SyncReader {
	device;
	static alphaModes = ["opaque", "premultiplied"];
	static width = 256;
	static height = 256;
	initialized = false;
	deviceStorage;
	deviceContexts;
	hostStorage;
	hostContext;
	constructor(device) {
		this.device = device;
	}
	#init() {
		if (typeof OffscreenCanvas === "undefined") throw new Error("OffscreenCanvas is not available in this environment, so you cannot read data from WebGPU synchronously. Consider using the async API.");
		const makeCanvas = () => new OffscreenCanvas(SyncReader.width, SyncReader.height);
		this.deviceStorage = SyncReader.alphaModes.map(makeCanvas);
		this.deviceContexts = this.deviceStorage.map((canvas, i) => {
			const context = canvas.getContext("webgpu");
			context.configure({
				device: this.device,
				format: "bgra8unorm",
				usage: GPUTextureUsage.COPY_DST,
				alphaMode: SyncReader.alphaModes[i]
			});
			return context;
		});
		this.hostStorage = makeCanvas();
		this.hostContext = this.hostStorage.getContext("2d", { willReadFrequently: true });
		this.initialized = true;
	}
	read(buffer, start, count) {
		if (!this.initialized) this.#init();
		const deviceStorage = this.deviceStorage;
		const deviceContexts = this.deviceContexts;
		const hostContext = this.hostContext;
		const pixelsSize = Math.ceil(count / 4);
		const bytesPerRow = SyncReader.width * 4;
		const valsGPU = /* @__PURE__ */ new ArrayBuffer(pixelsSize * 4);
		for (let i = 0; i < deviceContexts.length; i++) {
			const texture = deviceContexts[i].getCurrentTexture();
			const readData = (width, height, offset) => {
				const encoder = this.device.createCommandEncoder();
				encoder.copyBufferToTexture({
					buffer,
					bytesPerRow,
					offset: offset + start
				}, { texture }, {
					width,
					height,
					depthOrArrayLayers: 1
				});
				const commandBuffer = encoder.finish();
				this.device.queue.submit([commandBuffer]);
				hostContext.clearRect(0, 0, width, height);
				hostContext.drawImage(deviceStorage[i], 0, 0);
				const values = hostContext.getImageData(0, 0, width, height).data;
				const span = new Uint8ClampedArray(valsGPU, offset, 4 * width * height);
				const alphaMode = SyncReader.alphaModes[i];
				for (let k = 0; k < span.length; k += 4) if (alphaMode === "premultiplied") span[k + 3] = values[k + 3];
				else {
					span[k] = values[k + 2];
					span[k + 1] = values[k + 1];
					span[k + 2] = values[k];
				}
			};
			const pixelsPerCanvas = SyncReader.width * SyncReader.height;
			const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
			let remainder = pixelsSize % pixelsPerCanvas;
			const remainderRows = Math.floor(remainder / SyncReader.width);
			remainder = remainder % SyncReader.width;
			let offset = 0;
			for (let j = 0; j < wholeChunks; j++) {
				readData(SyncReader.width, SyncReader.height, offset);
				offset += pixelsPerCanvas * 4;
			}
			if (remainderRows > 0) {
				readData(SyncReader.width, remainderRows, offset);
				offset += remainderRows * SyncReader.width * 4;
			}
			if (remainder > 0) readData(remainder, 1, offset);
		}
		return new Uint8Array(valsGPU, 0, count);
	}
};
//#endregion
//#region src/backend/webgpu/routines.ts
function bitonicSortUniform(pass) {
	const ar = /* @__PURE__ */ new Uint32Array(3);
	ar[0] = pass.kind === "sort" ? 0 : 1;
	ar[1] = pass.mergeStep ?? 0;
	ar[2] = pass.mergeStage ?? 0;
	return new Uint8Array(ar.buffer);
}
/**
* Generate a bitonic sort shader.
*
* We implement a variant of bitonic sort that [only has forward comparators](
* <https://sortingalgos.miraheze.org/wiki/Bitonic_Sort#Bitonic_Sort_using_Forward_Comparators>),
* so we don't need to allocate memory for power-of-two padding.
*
* This uses workgroup shared memory up to `2*workgroupSize` elements, for each
* array in `batches`. For larger arrays, multiple passes are done:
*
* - Initial "sort" pass: each workgroup sorts its `2*workgroupSize` elements.
* - Subsequent "merge" passes: each pass merges sorted sequences of size
*   `2^(step+1)` with multiple workgroups. This doesn't use shared memory.
*
* The total number of passes is roughly `log2(n / workgroupSize)^2 / 2`.
*
* If `outputIndices` is true, the shader also tracks the original indices of
* the sorted elements (argsort) and outputs them to a separate buffer. This
* also makes the sorting algorithm stable.
*/
function bitonicSortShader(device, dtype, n, batches, outputIndices) {
	const ty = dtypeToWgsl(dtype, true);
	const paddedN = 1 << Math.ceil(Math.log2(n || 1));
	const numThreads = Math.ceil(paddedN / 2);
	const workgroupSize = require_backend.findPow2(numThreads, device.limits.maxComputeWorkgroupSizeX);
	const workgroupsPerBatch = numThreads / workgroupSize;
	const numStages = Math.log2(paddedN);
	const numLocalStages = Math.min(numStages, Math.log2(workgroupSize * 2));
	const needsF16 = dtype === "float16";
	const padValue = require_backend.isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype);
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;
${outputIndices ? `@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;` : ""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${ty}, ${workgroupSize * 2}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${workgroupSize * 2}>;` : ""}

fn compare(a: ${ty}, b: ${ty}) -> bool {
${require_backend.isFloatDtype(dtype) ? `
  let min_value = min(a, b);
  return a == min_value && b != min_value;` : "  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
${outputIndices ? `
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }` : `
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`}
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let batch = blockid / ${workgroupsPerBatch}u;
  let wg_in_batch = blockid % ${workgroupsPerBatch}u;

  let tid = local_id.x;
  let base = batch * ${n}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${numLocalStages - 1}u)) {
    let wg_base = wg_in_batch * ${workgroupSize * 2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${padValue}, input[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, input[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);` : ""}
    } else {
      shared_vals[idx0] = select(${padValue}, output[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, output[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${outputIndices ? `
      shared_idx[idx0] = select(${n}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_idx[idx1] = select(${n}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${n}u);` : ""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${numLocalStages - 1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${numLocalStages}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${n}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${outputIndices ? `output_idx[base + wg_base + idx0] = shared_idx[idx0];` : ""}
    }
    if (wg_base + idx1 < ${n}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${outputIndices ? `output_idx[base + wg_base + idx1] = shared_idx[idx1];` : ""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${workgroupSize} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${n}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
${outputIndices ? `
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;` : `
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`}
      }
    }
  }
}
`.trim();
	const grid = calculateGrid(batches * workgroupsPerBatch);
	const passes = [{ kind: "sort" }];
	for (let mergeStage = numLocalStages; mergeStage < numStages; mergeStage++) for (let mergeStep = mergeStage; mergeStep >= numLocalStages - 1; mergeStep--) passes.push({
		kind: "merge",
		mergeStep,
		mergeStage
	});
	return [{
		code,
		numInputs: 1,
		numOutputs: outputIndices ? 2 : 1,
		hasUniform: true,
		passes: passes.map((pass) => ({
			grid,
			uniform: bitonicSortUniform(pass)
		}))
	}];
}
function createSort(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	return bitonicSortShader(device, dtype, n, require_backend.prod(shape.slice(0, -1)), false);
}
function createArgsort(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	return bitonicSortShader(device, dtype, n, require_backend.prod(shape.slice(0, -1)), true);
}
function wgslCoordinate(linearIndex, view, axis) {
	if (view.shape[axis] <= 1) return "0u";
	const stride = view.strides[axis];
	return `(${stride === 1 ? linearIndex : `(${linearIndex} / ${stride}u)`} % ${view.shape[axis]}u)`;
}
function wgslBroadcastIndex(linearIndex, updateView, indexView, outDim, indexRank) {
	const firstUpdateDim = outDim + indexRank - indexView.ndim;
	const terms = [];
	for (let i = 0; i < indexView.ndim; i++) {
		const stride = indexView.strides[i];
		if (stride === 0) continue;
		const coord = wgslCoordinate(linearIndex, updateView, firstUpdateDim + i);
		terms.push(stride === 1 ? coord : `(${coord} * ${stride}u)`);
	}
	return terms.length === 0 ? "0u" : terms.join(" + ");
}
/**
* Generate the update statement for scatter-add.
*
* WGSL only has native integer atomics. Float32 uses a compare-exchange loop,
* while float16 updates one lane of a packed u32 with the same technique.
*/
function scatterAddWgsl(dtype) {
	switch (dtype) {
		case "float32": return {
			statement: "atomic_add_f32(&output[output_index], updates[global]);",
			helpers: `
fn atomic_add_f32(address: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var old_bits = atomicLoad(address);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`
		};
		case "float16": return {
			statement: `
  atomic_add_f16(
    &output[output_index / 2u],
    output_index % 2u != 0u,
    updates[global],
  );`.trim(),
			helpers: `
fn atomic_add_f16(
  address: ptr<storage, atomic<u32>, read_write>,
  high: bool,
  value: f16,
) {
  var old_bits = atomicLoad(address);
  loop {
    var pair = unpack2x16float(old_bits);
    if (high) {
      pair.y += f32(value);
    } else {
      pair.x += f32(value);
    }
    let new_bits = pack2x16float(pair);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`
		};
		case "uint32":
		case "int32": return {
			statement: "atomicAdd(&output[output_index], updates[global]);",
			helpers: ""
		};
		case "bool": return {
			statement: "atomicOr(&output[output_index], updates[global]);",
			helpers: ""
		};
		default: throw new Error(`Unsupported atomic Scatter dtype for WebGPU: ${dtype}`);
	}
}
function createScatter(device, type, { op, shape: outputShape, axis: indexedAxes, outDim, uniqueIndices }) {
	const dtype = type.inputDtypes[0];
	const updateView = require_backend.View.create(type.inputShapes[0]);
	const indexViews = type.inputShapes.slice(1).map((shape) => require_backend.View.create(shape));
	const indexRank = Math.max(...indexViews.map((view) => view.ndim));
	const updateCount = updateView.size;
	const elementType = dtypeToWgsl(dtype, true);
	const needsF16 = dtype === "float16";
	if (needsF16 && !device.features.has("shader-f16")) throw new Error("WebGPU device does not support shader-f16 feature");
	const freeUpdateDims = [...require_backend.range(outDim), ...require_backend.range(outDim + indexRank, updateView.ndim)];
	const indexLoads = indexViews.map((indexView, i) => {
		return `  let scatter_index_${i} = i32(index_${i}[${wgslBroadcastIndex("global", updateView, indexView, outDim, indexRank)}]);`;
	});
	const bounds = indexedAxes.map((outputAxis, i) => `scatter_index_${i} < 0 || scatter_index_${i} >= ${outputShape[outputAxis]}`);
	const outputView = require_backend.View.create(outputShape);
	let freeDim = 0;
	const outputIndex = require_backend.range(outputShape.length).map((outputAxis) => {
		const indexInput = indexedAxes.indexOf(outputAxis);
		let coord;
		if (indexInput === -1) coord = wgslCoordinate("global", updateView, freeUpdateDims[freeDim++]);
		else coord = `u32(scatter_index_${indexInput})`;
		return outputView.strides[outputAxis] === 1 ? coord : `(${coord} * ${outputView.strides[outputAxis]}u)`;
	}).join(" + ");
	const atomicAdd = op === "add" && !uniqueIndices;
	const outputStorageType = atomicAdd ? `atomic<${dtype === "int32" || dtype === "bool" ? "i32" : "u32"}>` : elementType;
	const updateSource = atomicAdd ? scatterAddWgsl(dtype) : {
		statement: "output[output_index] = updates[global];",
		helpers: ""
	};
	const maxThreads = device.limits.maxComputeWorkgroupSizeX;
	const workgroupSize = require_backend.findPow2(Math.min(Math.max(updateCount, 1), maxThreads), maxThreads);
	return [{
		code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}
${updateSource.helpers}

@group(0) @binding(0) var<storage, read> updates: array<${elementType}>;
${indexViews.map((_, i) => `@group(0) @binding(${i + 1}) var<storage, read> index_${i}: array<${dtypeToWgsl(type.inputDtypes[i + 1], true)}>;`).join("\n")}
@group(0) @binding(${type.inputShapes.length}) var<storage, read_write> output: array<${outputStorageType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
  if (global >= ${updateCount}u) {
    return;
  }
${indexLoads.join("\n")}
  if (${bounds.join(" || ")}) {
    return;
  }
  let output_index = ${outputIndex};
  ${updateSource.statement}
}
`.trim(),
		numInputs: type.inputShapes.length,
		numOutputs: 1,
		hasUniform: false,
		clearOutputs: true,
		passes: [{ grid: calculateGrid(Math.ceil(updateCount / workgroupSize)) }]
	}];
}
/**
* Generate a triangular solve shader.
*
* Solves A @ X.T = B.T for X, where A is upper-triangular.
* Uses a parallelized back-substitution:
*   1. Copy b to x
*   2. For j = n-1 down to 0:
*      - Divide x[j] by a[j,j] (single thread)
*      - All threads subtract x[j] * a[i,j] from x[i] for i < j in parallel
*/
function createTriangularSolve(device, type, params) {
	const dtype = type.inputDtypes[0];
	const aShape = type.inputShapes[0];
	const bShape = type.inputShapes[1];
	const n = aShape[aShape.length - 1];
	const numRhs = bShape[bShape.length - 2];
	const numMatrices = require_backend.prod(aShape.slice(0, -2));
	const needsF16 = dtype === "float16";
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = require_backend.findPow2(n, device.limits.maxComputeWorkgroupSizeX);
	return [{
		code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> a: array<${ty}>;
@group(0) @binding(1) var<storage, read> b: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> x: array<${ty}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let mat_idx = wg_idx / ${numRhs}u;
  let rhs_idx = wg_idx % ${numRhs}u;

  if (mat_idx >= ${numMatrices}u) {
    return;
  }

  let a_base = mat_idx * ${n * n}u;
  let bx_base = (mat_idx * ${numRhs}u + rhs_idx) * ${n}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${n}u; idx += ${workgroupSize}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${n}u; jj++) {
    let j = ${n - 1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${params.unitDiagonal ? `x_j = x[bx_base + j];` : `x_j = x[bx_base + j] / a[a_base + j * ${n}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${workgroupSize}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${n}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim(),
		numInputs: 2,
		numOutputs: 1,
		hasUniform: false,
		passes: [{ grid: calculateGrid(numMatrices * numRhs) }]
	}];
}
const CholeskyPhase = {
	Unblocked: 0,
	InitBlocked: 1,
	FactorBlock: 2,
	SolvePanel: 3,
	UpdateTrailing: 4
};
const CHOLESKY_BLOCK_SIZE = 16;
const CHOLESKY_BLOCK_THRESHOLD = 256;
const CHOLESKY_UPDATE_TILE_SIZE = 16;
function choleskyUniform(phase, k, blockSize, rowsBelow) {
	const ar = /* @__PURE__ */ new Uint32Array(4);
	ar[0] = phase;
	ar[1] = k;
	ar[2] = blockSize;
	ar[3] = rowsBelow;
	return new Uint8Array(ar.buffer);
}
/**
* Generate a Cholesky decomposition shader.
*
* Small matrices use the original one-pass Cholesky-Crout kernel. Larger
* matrices use a blocked multi-pass variant so the trailing update can run
* across many workgroups.
*/
function createCholesky(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = require_backend.prod(shape.slice(0, -2));
	const needsF16 = dtype === "float16";
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = Math.min(256, require_backend.findPow2(0, device.limits.maxComputeWorkgroupSizeX));
	const useBlocked = n >= CHOLESKY_BLOCK_THRESHOLD;
	const blockSize = useBlocked ? CHOLESKY_BLOCK_SIZE : n;
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;

struct CholeskyParams {
  phase: u32,
  k: u32,
  block_size: u32,
  rows_below: u32,
}

@group(1) @binding(0) var<uniform> params: CholeskyParams;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let tid = local_id.x;

  if (params.phase == ${CholeskyPhase.Unblocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    // Zero out output and copy lower triangle from input.
    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    storageBarrier();

    // Cholesky-Crout algorithm: process column by column.
    for (var j = 0u; j < ${n}u; j++) {
      for (var i = j + tid; i < ${n}u; i += ${workgroupSize}u) {
        var sum = output[mat_idx(base, i, j)];
        for (var k = 0u; k < j; k++) {
          sum -= output[mat_idx(base, i, k)] * output[mat_idx(base, j, k)];
        }
        output[mat_idx(base, i, j)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, j, j)]);
        output[mat_idx(base, j, j)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < ${n}u; i += ${workgroupSize}u) {
        output[mat_idx(base, i, j)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.InitBlocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.FactorBlock}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;
    let k0 = params.k;
    let b = params.block_size;

    for (var j = 0u; j < b; j++) {
      let col = k0 + j;
      for (var i = j + tid; i < b; i += ${workgroupSize}u) {
        let row = k0 + i;
        var sum = output[mat_idx(base, row, col)];
        for (var r = 0u; r < j; r++) {
          sum -= output[mat_idx(base, row, k0 + r)] *
            output[mat_idx(base, col, k0 + r)];
        }
        output[mat_idx(base, row, col)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, col, col)]);
        output[mat_idx(base, col, col)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < b; i += ${workgroupSize}u) {
        output[mat_idx(base, k0 + i, col)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.SolvePanel}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
    if (global >= rows * ${batches}u) {
      return;
    }

    let batch = global / rows;
    let row = params.k + params.block_size + (global % rows);
    let base = batch * ${n * n}u;

    for (var j = 0u; j < params.block_size; j++) {
      let col = params.k + j;
      var sum = output[mat_idx(base, row, col)];
      for (var r = 0u; r < j; r++) {
        sum -= output[mat_idx(base, row, params.k + r)] *
          output[mat_idx(base, col, params.k + r)];
      }
      output[mat_idx(base, row, col)] =
        sum / output[mat_idx(base, col, col)];
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.UpdateTrailing}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let tile_count = (rows + ${CHOLESKY_UPDATE_TILE_SIZE - 1}u) / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let tiles_per_batch = tile_count * (tile_count + 1u) / 2u;
    let wg_global = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (wg_global >= tiles_per_batch * ${batches}u) {
      return;
    }

    let batch = wg_global / tiles_per_batch;
    let tile = wg_global % tiles_per_batch;
    var tile_row = u32((sqrt(f32(8u * tile + 1u)) - 1.0) * 0.5);
    // Correct for GPU sqrt rounding around exact triangular-number boundaries.
    var row_start = tile_row * (tile_row + 1u) / 2u;
    if (row_start > tile) {
      tile_row -= 1u;
      row_start = tile_row * (tile_row + 1u) / 2u;
    }
    let next_row_start = (tile_row + 1u) * (tile_row + 2u) / 2u;
    if (tile >= next_row_start) {
      tile_row += 1u;
      row_start = next_row_start;
    }
    let tile_col = tile - row_start;
    let local_row = tid / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let local_col = tid % ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let row_local = tile_row * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_row;
    let col_local = tile_col * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_col;
    if (row_local >= rows || col_local >= rows) {
      return;
    }
    if (row_local < col_local) {
      return;
    }

    let row = params.k + params.block_size + row_local;
    let col = params.k + params.block_size + col_local;
    let base = batch * ${n * n}u;
    var sum = output[mat_idx(base, row, col)];
    for (var r = 0u; r < params.block_size; r++) {
      sum -= output[mat_idx(base, row, params.k + r)] *
        output[mat_idx(base, col, params.k + r)];
    }
    output[mat_idx(base, row, col)] = sum;
  }
}
`.trim();
	const passes = [];
	if (!useBlocked) passes.push({
		grid: calculateGrid(batches),
		uniform: choleskyUniform(CholeskyPhase.Unblocked, 0, n, 0)
	});
	else {
		passes.push({
			grid: calculateGrid(batches),
			uniform: choleskyUniform(CholeskyPhase.InitBlocked, 0, blockSize, 0)
		});
		for (let k = 0; k < n; k += blockSize) {
			const b = Math.min(blockSize, n - k);
			const rowsBelow = n - k - b;
			passes.push({
				grid: calculateGrid(batches),
				uniform: choleskyUniform(CholeskyPhase.FactorBlock, k, b, rowsBelow)
			});
			if (rowsBelow > 0) {
				passes.push({
					grid: calculateGrid(Math.ceil(rowsBelow * batches / workgroupSize)),
					uniform: choleskyUniform(CholeskyPhase.SolvePanel, k, b, rowsBelow)
				});
				passes.push({
					grid: calculateGrid(Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) * (Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) + 1) * batches / 2),
					uniform: choleskyUniform(CholeskyPhase.UpdateTrailing, k, b, rowsBelow)
				});
			}
		}
	}
	return [{
		code,
		numInputs: 1,
		numOutputs: 1,
		hasUniform: true,
		passes
	}];
}
/**
* Generate an LU decomposition shader with partial pivoting.
*
* Computes PA = LU where P is a permutation matrix, L is lower triangular
* with unit diagonal, and U is upper triangular.
*
* For each column j:
*   1. Find pivot row (max absolute value in column j, rows >= j)
*   2. Swap rows j and pivot row
*   3. Compute L[i][j] = A[i][j] / A[j][j] for i > j
*   4. Update submatrix: A[i][k] -= L[i][j] * A[j][k] for i > j, k > j
*/
function createLU(device, type) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const m = shape[shape.length - 2];
	const n = shape[shape.length - 1];
	const r = Math.min(m, n);
	const batches = require_backend.prod(shape.slice(0, -2));
	const needsF16 = dtype === "float16";
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = require_backend.findPow2(Math.max(m, n), device.limits.maxComputeWorkgroupSizeX);
	return [{
		code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let lu_base = batch * ${m * n}u;
  let piv_base = batch * ${r}u;
  let perm_base = batch * ${m}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${m * n}u; idx += ${workgroupSize}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${m}u; idx += ${workgroupSize}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${r}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${n}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${m}u; i++) {
        let val = abs(lu[lu_base + i * ${n}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${n}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${n}u; col += ${workgroupSize}u) {
        let tmp = lu[lu_base + j * ${n}u + col];
        lu[lu_base + j * ${n}u + col] = lu[lu_base + pr * ${n}u + col];
        lu[lu_base + pr * ${n}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${m}u; i += ${workgroupSize}u) {
      let factor = lu[lu_base + i * ${n}u + j] / pivot_val;
      lu[lu_base + i * ${n}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${n}u; k++) {
        lu[lu_base + i * ${n}u + k] -= factor * lu[lu_base + j * ${n}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim(),
		numInputs: 1,
		numOutputs: 3,
		hasUniform: false,
		passes: [{ grid: calculateGrid(batches) }]
	}];
}
function createJacobiEigh(device, type, params) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = require_backend.prod(shape.slice(0, -2));
	const needsF16 = dtype === "float16";
	const ty = dtypeToWgsl(dtype, true);
	const tolerance = `${ty}(${params.tolerance})`;
	const workgroupSize = require_backend.findPow2(Math.max(n, 1), device.limits.maxComputeWorkgroupSizeX);
	return [{
		code: `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> diagonalized: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> vectors: array<${ty}>;

var<workgroup> done: u32;
var<workgroup> rot_active: u32;
var<workgroup> rot_c: ${ty};
var<workgroup> rot_s: ${ty};
var<workgroup> rot_app: ${ty};
var<workgroup> rot_aqq: ${ty};
var<workgroup> rot_apq: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

fn sym_idx(base: u32, row: u32, col: u32) -> u32 {
  return mat_idx(base, max(row, col), min(row, col));
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let base = batch * ${n * n}u;
  let tid = local_id.x;

  for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
    let row = idx / ${n}u;
    let col = idx % ${n}u;
    diagonalized[base + idx] = select(
      ${ty}(0),
      input[base + idx],
      row >= col,
    );
    vectors[base + idx] = select(${ty}(0), ${ty}(1), row == col);
  }
  storageBarrier();

  for (var sweep = 0u; sweep < ${params.maxSweeps}u; sweep++) {
    if (tid == 0u) {
      var max_abs = ${ty}(1);
      var max_offdiag = ${ty}(0);
      for (var idx = 0u; idx < ${n * n}u; idx++) {
        let row = idx / ${n}u;
        let col = idx % ${n}u;
        let value = abs(diagonalized[base + idx]);
        max_abs = max(max_abs, value);
        if (row > col) {
          max_offdiag = max(max_offdiag, value);
        }
      }
      done = select(0u, 1u, max_offdiag <= ${tolerance} * max_abs);
    }
    let done_uniform = workgroupUniformLoad(&done);
    if (done_uniform != 0u) {
      break;
    }

    for (var p = 0u; p + 1u < ${n}u; p++) {
      for (var q = p + 1u; q < ${n}u; q++) {
        if (tid == 0u) {
          rot_app = diagonalized[mat_idx(base, p, p)];
          rot_aqq = diagonalized[mat_idx(base, q, q)];
          rot_apq = diagonalized[sym_idx(base, p, q)];
          if (rot_apq == ${ty}(0)) {
            rot_active = 0u;
            rot_c = ${ty}(1);
            rot_s = ${ty}(0);
          } else {
            let tau = (rot_aqq - rot_app) / (${ty}(2) * rot_apq);
            let tau_sign = select(${ty}(-1), ${ty}(1), tau >= ${ty}(0));
            let t = tau_sign / (abs(tau) + sqrt(tau * tau + ${ty}(1)));
            rot_c = inverseSqrt(t * t + ${ty}(1));
            rot_s = t * rot_c;
            rot_active = 1u;
          }
        }
        workgroupBarrier();

        if (rot_active != 0u) {
          for (var k = tid; k < ${n}u; k += ${workgroupSize}u) {
            if (k != p && k != q) {
              let kp = sym_idx(base, k, p);
              let kq = sym_idx(base, k, q);
              let akp = diagonalized[kp];
              let akq = diagonalized[kq];
              let next_kp = rot_c * akp - rot_s * akq;
              let next_kq = rot_s * akp + rot_c * akq;
              diagonalized[kp] = next_kp;
              diagonalized[kq] = next_kq;
            } else if (k == p) {
              diagonalized[mat_idx(base, p, p)] =
                rot_c * rot_c * rot_app - ${ty}(2) * rot_s * rot_c * rot_apq + rot_s * rot_s * rot_aqq;
              diagonalized[sym_idx(base, p, q)] = ${ty}(0);
            } else {
              diagonalized[mat_idx(base, q, q)] =
                rot_s * rot_s * rot_app + ${ty}(2) * rot_s * rot_c * rot_apq + rot_c * rot_c * rot_aqq;
            }

            let vp = mat_idx(base, k, p);
            let vq = mat_idx(base, k, q);
            let vkp = vectors[vp];
            let vkq = vectors[vq];
            vectors[vp] = rot_c * vkp - rot_s * vkq;
            vectors[vq] = rot_s * vkp + rot_c * vkq;
          }
        }
        storageBarrier();
      }
    }
  }
}
`.trim(),
		numInputs: 1,
		numOutputs: 2,
		hasUniform: false,
		passes: [{ grid: calculateGrid(batches) }]
	}];
}
function fftUniform(phase, radix, prev, normalize) {
	return new Uint8Array(new Uint32Array([
		phase,
		radix,
		prev,
		normalize ? 1 : 0
	]).buffer);
}
function createFft(device, type, params) {
	const dtype = type.inputDtypes[0];
	const shape = type.inputShapes[0];
	const n = shape[shape.length - 1];
	const batches = require_backend.prod(shape.slice(0, -1));
	if (require_backend.prod(params.factors) !== n) throw new Error(`fft: factorization ${params.factors} does not match size ${n}`);
	const needsF16 = dtype === "float16";
	const ty = dtypeToWgsl(dtype, true);
	const workgroupSize = Math.min(256, require_backend.findPow2(0, device.limits.maxComputeWorkgroupSizeX));
	const maxFactor = Math.max(1, ...params.factors);
	const angleScale = params.inverse ? "6.283185307179586" : "-6.283185307179586";
	const digitReversal = params.factors.map((factor) => `
  digit = remaining % ${factor}u;
  remaining = remaining / ${factor}u;
  stride = stride * ${factor}u;
  reversed = reversed + digit * (${n}u / stride);`).join("");
	const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input_real: array<${ty}>;
@group(0) @binding(1) var<storage, read> input_imag: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> output_real: array<${ty}>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<${ty}>;

struct FftParams {
  phase: u32,
  radix: u32,
  prev: u32,
  normalize: u32,
}

@group(1) @binding(0) var<uniform> fft_params: FftParams;

fn digit_reversed_index(index: u32) -> u32 {
  var remaining = index;
  var stride = 1u;
  var reversed = 0u;
  var digit = 0u;
${digitReversal}
  return reversed;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;

  if (fft_params.phase == 0u) {
    if (global >= ${batches * n}u) {
      return;
    }
    let batch = global / ${n}u;
    let out_idx = global % ${n}u;
    let source = batch * ${n}u + digit_reversed_index(out_idx);
    output_real[global] = input_real[source];
    output_imag[global] = input_imag[source];
    return;
  }

  let butterflies_per_batch = ${n}u / fft_params.radix;
  if (global >= ${batches}u * butterflies_per_batch) {
    return;
  }

  let batch = global / butterflies_per_batch;
  let local = global % butterflies_per_batch;
  let j = local % fft_params.prev;
  let group = local / fft_params.prev;
  let span = fft_params.prev * fft_params.radix;
  let start = batch * ${n}u + group * span + j;
  let scale = select(1.0, 1.0 / f32(${n}u), fft_params.normalize != 0u);

  var scratch_real: array<f32, ${maxFactor}>;
  var scratch_imag: array<f32, ${maxFactor}>;

  for (var q = 0u; q < fft_params.radix; q++) {
    let idx = start + q * fft_params.prev;
    let angle = ${angleScale} * f32(q * j) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let xr = f32(output_real[idx]);
    let xi = f32(output_imag[idx]);
    scratch_real[q] = xr * c - xi * s;
    scratch_imag[q] = xr * s + xi * c;
  }

  for (var p = 0u; p < fft_params.radix; p++) {
    var sum_real = 0.0;
    var sum_imag = 0.0;
    for (var q = 0u; q < fft_params.radix; q++) {
      let angle = ${angleScale} * f32(q * p) / f32(fft_params.radix);
      let c = cos(angle);
      let s = sin(angle);
      let xr = scratch_real[q];
      let xi = scratch_imag[q];
      sum_real += xr * c - xi * s;
      sum_imag += xr * s + xi * c;
    }
    let idx = start + p * fft_params.prev;
    output_real[idx] = ${ty}(sum_real * scale);
    output_imag[idx] = ${ty}(sum_imag * scale);
  }
}
`.trim();
	const passes = [{
		grid: calculateGrid(Math.ceil(batches * n / workgroupSize)),
		uniform: fftUniform(0, 1, 1, false)
	}];
	let prev = 1;
	for (let i = 0; i < params.factors.length; i++) {
		const radix = params.factors[i];
		passes.push({
			grid: calculateGrid(Math.ceil(batches * n / radix / workgroupSize)),
			uniform: fftUniform(1, radix, prev, params.inverse && i === params.factors.length - 1)
		});
		prev *= radix;
	}
	return [{
		code,
		numInputs: 2,
		numOutputs: 2,
		hasUniform: true,
		passes
	}];
}
function createRoutineShader(device, routine) {
	switch (routine.name) {
		case "Sort": return createSort(device, routine.type);
		case "Argsort": return createArgsort(device, routine.type);
		case "Scatter": return createScatter(device, routine.type, routine.params);
		case "TriangularSolve": return createTriangularSolve(device, routine.type, routine.params);
		case "Cholesky": return createCholesky(device, routine.type);
		case "LU": return createLU(device, routine.type);
		case "JacobiEigh": return createJacobiEigh(device, routine.type, routine.params);
		case "Fft": return createFft(device, routine.type, routine.params);
		default: throw new require_backend.UnsupportedRoutineError(routine.name, "webgpu");
	}
}
//#endregion
//#region src/backend/webgpu/tracing.ts
const MAX_TIMESTAMP_QUERIES = 4096;
const activeBatch = /* @__PURE__ */ new WeakMap();
function createTracingBatch(device) {
	return {
		querySet: device.createQuerySet({
			type: "timestamp",
			count: MAX_TIMESTAMP_QUERIES
		}),
		resolve: device.createBuffer({
			size: MAX_TIMESTAMP_QUERIES * 8,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
		}),
		dst: device.createBuffer({
			size: MAX_TIMESTAMP_QUERIES * 8,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		}),
		nextIndex: 0,
		entries: []
	};
}
function acquireTracingSlot(device) {
	if (!device.features.has("timestamp-query")) return void 0;
	let batch = activeBatch.get(device);
	if (batch && batch.nextIndex >= MAX_TIMESTAMP_QUERIES) {
		flushTracingBatch(device, batch);
		batch = void 0;
	}
	if (!batch) {
		batch = createTracingBatch(device);
		activeBatch.set(device, batch);
		require_backend.onFlushTrace(() => {
			const b = activeBatch.get(device);
			if (b && b.entries.length > 0) flushTracingBatch(device, b);
			activeBatch.delete(device);
		});
	}
	const beginIndex = batch.nextIndex;
	const endIndex = beginIndex + 1;
	batch.nextIndex += 2;
	return {
		batch,
		beginIndex,
		endIndex
	};
}
/**
* If tracing is active, acquire a slot for timestamp queries.
*
* Returns undefined if tracing is not active or the device doesn't support
* timestamp queries.
*/
function maybeAcquireTracingSlot(device) {
	if (!require_backend.isTracing()) return void 0;
	return acquireTracingSlot(device);
}
/**
* Record a tracing entry for a pipeline dispatch and schedule an auto-flush.
*/
function recordTrace(device, slot, source, numPasses, wgslSource) {
	const info = require_backend.traceSourceInfo(source);
	info.properties.push(["passes", `${numPasses}`]);
	info.properties.push(["source", wgslSource]);
	slot.batch.entries.push({
		...info,
		beginIndex: slot.beginIndex,
		endIndex: slot.endIndex
	});
	scheduleAutoFlush(device);
}
/**
* If the active batch has pending entries, flush and replace it so traces
* are emitted without waiting for the batch to fill or stopTrace().
*
* Called after each dispatch records its entry via a microtask so that
* synchronous back-to-back dispatches are still batched together.
*/
function scheduleAutoFlush(device) {
	queueMicrotask(() => {
		const batch = activeBatch.get(device);
		if (batch && batch.entries.length > 0) {
			flushTracingBatch(device, batch);
			activeBatch.set(device, createTracingBatch(device));
		}
	});
}
function flushTracingBatch(device, batch) {
	if (batch.entries.length === 0) return;
	const usedQueries = batch.nextIndex;
	const encoder = device.createCommandEncoder();
	encoder.resolveQuerySet(batch.querySet, 0, usedQueries, batch.resolve, 0);
	encoder.copyBufferToBuffer(batch.resolve, 0, batch.dst, 0, usedQueries * 8);
	device.queue.submit([encoder.finish()]);
	const { entries } = batch;
	batch.dst.mapAsync(GPUMapMode.READ).then(() => {
		try {
			const times = new BigInt64Array(batch.dst.getMappedRange());
			const anchorGpuNs = times[entries[entries.length - 1].endIndex];
			const anchorCpuMs = performance.now();
			for (const entry of entries) require_backend.emitTrace("webgpu", entry, anchorCpuMs + Number(times[entry.beginIndex] - anchorGpuNs) / 1e6, anchorCpuMs + Number(times[entry.endIndex] - anchorGpuNs) / 1e6);
		} finally {
			batch.dst.unmap();
			batch.querySet.destroy();
			batch.resolve.destroy();
			batch.dst.destroy();
		}
	});
}
//#endregion
//#region src/backend/webgpu.ts
const MAX_REUSABLE_BUFFER_BYTES = 64 * 1024 * 1024;
const MAX_REUSABLE_BUFFERS_PER_SIZE = 64;
/** Implementation of `Backend` that uses WebGPU in browsers. */
var WebGPUBackend = class {
	device;
	type = "webgpu";
	maxArgs;
	pipelines;
	syncReader;
	buffers;
	nextSlot;
	#cachedShaderMap = /* @__PURE__ */ new Map();
	#reusableZsb;
	#bufferPool = /* @__PURE__ */ new Map();
	constructor(device) {
		this.device = device;
		if (require_backend.DEBUG >= 3 && device.adapterInfo) console.info("webgpu adapter:", device.adapterInfo.vendor, device.adapterInfo.architecture);
		this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
		this.pipelines = new ShaderPipelineCache(device);
		this.syncReader = new SyncReader(device);
		this.buffers = /* @__PURE__ */ new Map();
		this.nextSlot = 1;
		this.#reusableZsb = this.#createBuffer(4);
		device.addEventListener("uncapturederror", (event) => {
			console.error("Uncaptured error in WebGPU backend:", event.error.message);
		});
	}
	malloc(size, initialData) {
		if (initialData && initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
		const allocatedSize = Math.ceil(size / 4) * 4 || 4;
		const buffer = size === 0 ? this.#reusableZsb : this.#acquireBuffer(allocatedSize);
		if (initialData && size > 0) if (initialData.byteLength % 4 === 0) this.device.queue.writeBuffer(buffer, 0, initialData);
		else {
			const aligned = initialData.byteLength - initialData.byteLength % 4;
			if (aligned > 0) this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
			const remainder = /* @__PURE__ */ new Uint8Array(4);
			remainder.set(initialData.subarray(aligned));
			this.device.queue.writeBuffer(buffer, aligned, remainder);
		}
		const slot = this.nextSlot++;
		this.buffers.set(slot, {
			buffer,
			size,
			allocatedSize,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new require_backend.SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new require_backend.SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) {
			this.buffers.delete(slot);
			if (buffer.buffer !== this.#reusableZsb) this.#releaseBuffer(buffer.buffer, buffer.allocatedSize);
		}
	}
	async read(slot, start, count) {
		const { buffer, size } = this.#getBuffer(slot);
		if (buffer === this.#reusableZsb) return /* @__PURE__ */ new Uint8Array();
		if (start === void 0) start = 0;
		if (count === void 0) count = size - start;
		const paddedSize = Math.ceil(count / 4) * 4;
		const staging = this.#createBuffer(paddedSize, { read: true });
		try {
			const commandEncoder = this.device.createCommandEncoder();
			commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
			this.device.queue.submit([commandEncoder.finish()]);
			await mapAsyncRead(this.device, staging);
			const arrayBuffer = staging.getMappedRange();
			return new Uint8Array(arrayBuffer.slice(), 0, count);
		} finally {
			staging.destroy();
		}
	}
	readSync(slot, start, count) {
		const { buffer, size } = this.#getBuffer(slot);
		if (buffer === this.#reusableZsb) return /* @__PURE__ */ new Uint8Array();
		if (start === void 0) start = 0;
		if (count === void 0) count = size - start;
		return this.syncReader.read(buffer, start, count);
	}
	#cachedShader(kernel) {
		const cacheKey = require_backend.FpHash.hash(kernel);
		let result = this.#cachedShaderMap.get(cacheKey);
		if (!result) {
			result = pipelineSource(this.device, kernel);
			this.#cachedShaderMap.set(cacheKey, result);
		}
		return result;
	}
	async prepareKernel(kernel) {
		const shader = this.#cachedShader(kernel);
		const pipeline = await this.pipelines.prepare(shader);
		return new require_backend.Executable(kernel, [{
			...shader,
			pipeline
		}]);
	}
	prepareKernelSync(kernel) {
		const shader = this.#cachedShader(kernel);
		const pipeline = this.pipelines.prepareSync(shader);
		return new require_backend.Executable(kernel, [{
			...shader,
			pipeline
		}]);
	}
	async prepareRoutine(routine) {
		const shaders = createRoutineShader(this.device, routine);
		return new require_backend.Executable(routine, await Promise.all(shaders.map(async (shader) => {
			const pipeline = await this.pipelines.prepare(shader);
			return {
				...shader,
				pipeline
			};
		})));
	}
	prepareRoutineSync(routine) {
		return new require_backend.Executable(routine, createRoutineShader(this.device, routine).map((shader) => {
			const pipeline = this.pipelines.prepareSync(shader);
			return {
				...shader,
				pipeline
			};
		}));
	}
	dispatch(exe, inputs, outputs) {
		const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
		const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
		pipelineSubmit(this.device, exe, inputBuffers, outputBuffers);
	}
	#getBuffer(slot) {
		const buffer = this.buffers.get(slot);
		if (!buffer) throw new require_backend.SlotError(slot);
		return {
			buffer: buffer.buffer,
			size: buffer.size
		};
	}
	#acquireBuffer(size) {
		if (size > MAX_REUSABLE_BUFFER_BYTES) return this.#createBuffer(size);
		const bucket = this.#bufferPool.get(size);
		const buffer = bucket?.pop();
		if (bucket && bucket.length === 0) this.#bufferPool.delete(size);
		return buffer ?? this.#createBuffer(size);
	}
	#releaseBuffer(buffer, size) {
		if (size > MAX_REUSABLE_BUFFER_BYTES) {
			buffer.destroy();
			return;
		}
		const bucket = this.#bufferPool.get(size);
		if (!bucket) {
			this.#bufferPool.set(size, [buffer]);
			return;
		}
		if (bucket.length >= MAX_REUSABLE_BUFFERS_PER_SIZE) {
			buffer.destroy();
			return;
		}
		bucket.push(buffer);
	}
	/**
	* Create a GPU buffer.
	*
	* By default, this creates a general-purpose buffer with the given size.
	*
	* - If `mapped` is true, initialize the buffer in mapped mode so that it can
	*   be populated with data from the CPU. (Call `.unmap()` later.)
	* - If `read` is true, create a staging buffer for returning data to CPU.
	*   (Call `.mapAsync()` later.)
	*/
	#createBuffer(size, { mapped = false, read = false } = {}) {
		if (read && mapped) throw new Error("mapped and read cannot both be true");
		return this.device.createBuffer({
			size,
			usage: read ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			mappedAtCreation: mapped
		});
	}
};
/**
* Compiles an expression into WebGPU shader source code.
*
* Returns the shader source and the number of workgroups to dispatch along x
* and y axes, to run the kernel.
*/
function pipelineSource(device, kernel) {
	const nullaryKernel = nullaryKernelSource(device, kernel);
	if (nullaryKernel) return nullaryKernel;
	const tune = require_backend.tuneWebgpu(kernel);
	if (require_backend.DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
	const { nargs, reduction: re } = kernel;
	const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
	const wb = new WgslBuilder();
	wb.emitPreamble(device, [tune.exp, tune.epilogue]);
	const usedArgs = Array.from({ length: nargs }, () => null);
	tune.exp.fold((exp) => {
		if (exp.op === "GlobalIndex") usedArgs[exp.arg[0]] = exp.dtype;
	});
	tune.epilogue?.fold((exp) => {
		if (exp.op === "GlobalIndex") usedArgs[exp.arg[0]] = exp.dtype;
	});
	for (let i = 0; i < nargs; i++) {
		const ty = dtypeToWgsl(usedArgs[i] ?? "float32", true);
		wb.emit(`@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`);
	}
	const resultTy = dtypeToWgsl(kernel.dtype, true);
	wb.emit(`@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`);
	const groupCount = re ? tune.size.groups ?? 1 : 1;
	const groupedReduction = re && groupCount > 1;
	if (groupedReduction && tune.threadCount % groupCount !== 0) throw new Error("WebGPU grouped reduction has invalid thread count");
	if (groupedReduction && groupCount > device.limits.maxComputeWorkgroupSizeX) throw new Error("WebGPU grouped reduction exceeds workgroup size limit");
	const workgroupSize = groupedReduction ? groupCount : require_backend.findPow2(tune.threadCount, 256);
	const gridSize = groupedReduction ? tune.threadCount / groupCount : Math.ceil(tune.threadCount / workgroupSize);
	const [gridX, gridY] = calculateGrid(gridSize);
	if (groupedReduction) {
		const partialTy = dtypeToWgsl(re.dtype);
		for (let i = 0; i < (tune.size.upcast ?? 1); i++) wb.emit(`var<workgroup> partial${i}: array<${partialTy}, ${groupCount}>;`);
	}
	wb.emit("", `@compute @workgroup_size(${workgroupSize})`);
	if (groupedReduction) {
		wb.emit("fn main(", wb.pushIndent, "@builtin(local_invocation_id) lid : vec3<u32>,", "@builtin(workgroup_id) wg_id : vec3<u32>,", wb.popIndent, ") {", wb.pushIndent);
		if (gridY === 1) wb.emit(`if (wg_id.x >= ${gridSize}u) { return; }`, "let gidx: i32 = i32(wg_id.x);");
		else wb.emit(`if (${gridX}u * wg_id.y + wg_id.x >= ${gridSize}u) { return; }`, `let gidx: i32 = i32(${gridX}u * wg_id.y + wg_id.x);`);
		wb.emit("let group: i32 = i32(lid.x);");
	} else {
		wb.emit("fn main(@builtin(global_invocation_id) id : vec3<u32>) {", wb.pushIndent);
		if (gridY === 1) wb.emit(`if (id.x >= ${tune.threadCount}) { return; }`, "let gidx: i32 = i32(id.x);");
		else {
			const sizeX = gridX * workgroupSize;
			wb.emit(`if (${sizeX} * id.y + id.x >= ${tune.threadCount}) { return; }`, `let gidx: i32 = i32(${sizeX} * id.y + id.x);`);
		}
	}
	wb.emitPhonyAssignments(args);
	const gen = new WgslExpCodegen(wb, args);
	if (!re) {
		gen.countReferences(tune.exp);
		let rhs = require_backend.strip1(gen.run(tune.exp));
		if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
		wb.emit(`result[gidx] = ${rhs};`);
	} else {
		const unroll = tune.size.unroll ?? 1;
		const upcast = tune.size.upcast ?? 1;
		const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
		for (let i = 0; i < upcast; i++) wb.emit(`var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`);
		wb.emit(`for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`, wb.pushIndent);
		const exps = [];
		const cache = /* @__PURE__ */ new Map();
		for (let up = 0; up < upcast; up++) {
			exps.push([]);
			for (let un = 0; un < unroll; un++) {
				const exp = tune.exp.substitute({
					upcast: require_backend.AluExp.i32(up),
					unroll: require_backend.AluExp.i32(un)
				});
				exps[up].push(exp.simplify(cache));
				gen.countReferences(exps[up][un]);
			}
		}
		const items = exps.map((ar) => ar.map((x) => gen.run(x)).map(require_backend.strip1));
		for (let i = 0; i < upcast; i++) {
			let rhs = items[i][0];
			for (let j = 1; j < unroll; j++) if (re.op === "Add") rhs = `${rhs} + ${items[i][j]}`;
			else if (re.op === "Mul") rhs = `${rhs} * ${items[i][j]}`;
			else if (re.op === "Min") rhs = re.dtype === "bool" ? `(${rhs} && ${items[i][j]})` : `min(${rhs}, ${items[i][j]})`;
			else if (re.op === "Max") rhs = re.dtype === "bool" ? `(${rhs} || ${items[i][j]})` : `max(${rhs}, ${items[i][j]})`;
			else throw new Error(`Unsupported reduction op: ${re.op}`);
			if (re.op === "Add") wb.emit(`${acc[i]} += ${rhs};`);
			else if (re.op === "Mul") wb.emit(`${acc[i]} *= ${rhs};`);
			else if (re.op === "Min") if (re.dtype === "bool") wb.emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
			else wb.emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
			else if (re.op === "Max") if (re.dtype === "bool") wb.emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
			else wb.emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
			else throw new Error(`Unsupported reduction op: ${re.op}`);
		}
		wb.emit(wb.popIndent, "}");
		if (groupedReduction) {
			for (let i = 0; i < upcast; i++) wb.emit(`partial${i}[lid.x] = ${acc[i]};`);
			wb.emit("workgroupBarrier();");
			for (let stride = groupCount / 2; stride >= 1; stride /= 2) {
				wb.emit(`if (lid.x < ${stride}u) {`, wb.pushIndent);
				for (let i = 0; i < upcast; i++) wb.emit(`partial${i}[lid.x] = ${reduceOpWgsl(re.op, re.dtype, `partial${i}[lid.x]`, `partial${i}[lid.x + ${stride}u]`)};`);
				wb.emit(wb.popIndent, "}", "workgroupBarrier();");
			}
		}
		gen.reset();
		const outputIdxExps = [];
		const fusionExps = [];
		for (let i = 0; i < upcast; i++) {
			const exp = tune.outputIdxExp.substitute({ upcast: require_backend.AluExp.i32(i) });
			outputIdxExps.push(exp.simplify(cache));
			gen.countReferences(outputIdxExps[i]);
			fusionExps.push(tune.epilogue.substitute({
				acc: require_backend.AluExp.variable(re.dtype, acc[i]),
				upcast: require_backend.AluExp.i32(i)
			}).simplify(cache));
			gen.countReferences(fusionExps[i]);
		}
		if (groupedReduction) {
			wb.emit("if (lid.x == 0u) {", wb.pushIndent);
			for (let i = 0; i < upcast; i++) wb.emit(`${acc[i]} = partial${i}[0u];`);
		}
		for (let i = 0; i < upcast; i++) {
			const index = require_backend.strip1(gen.run(outputIdxExps[i]));
			let rhs = require_backend.strip1(gen.run(fusionExps[i]));
			if (resultTy !== dtypeToWgsl(fusionExps[i].dtype)) rhs = `${resultTy}(${rhs})`;
			wb.emit(`result[${index}] = ${rhs};`);
		}
		if (groupedReduction) wb.emit(wb.popIndent, "}");
	}
	wb.emit(wb.popIndent, "}");
	return {
		code: wb.toString(),
		numInputs: nargs,
		numOutputs: 1,
		hasUniform: false,
		passes: [{ grid: [gridX, gridY] }]
	};
}
function pipelineSubmit(device, exe, inputs, outputs) {
	const { data: pipelines, source } = exe;
	const commandEncoder = device.createCommandEncoder();
	for (const { pipeline, ...shader } of pipelines) {
		if (inputs.length !== shader.numInputs || outputs.length !== shader.numOutputs) throw new Error(`webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, got ${inputs.length} inputs and ${outputs.length} outputs`);
		if (shader.clearOutputs) for (const output of outputs) commandEncoder.clearBuffer(output);
		const filteredPasses = shader.passes.filter(({ grid }) => require_backend.prod(grid) > 0);
		if (filteredPasses.length === 0) continue;
		const slot = maybeAcquireTracingSlot(device);
		const bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [...inputs.map((buffer, i) => ({
				binding: i,
				resource: { buffer }
			})), ...outputs.map((buffer, i) => ({
				binding: inputs.length + i,
				resource: { buffer }
			}))]
		});
		let uniformBindGroup = null;
		let uniformAlignment = 0;
		if (shader.hasUniform) {
			const [uniformBuffer, alignment] = combineUniforms(device, filteredPasses.map(({ uniform }) => uniform));
			uniformAlignment = alignment;
			uniformBindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(1),
				entries: [{
					binding: 0,
					resource: {
						buffer: uniformBuffer,
						size: alignment
					}
				}]
			});
		}
		for (let i = 0; i < filteredPasses.length; i++) {
			const { grid } = filteredPasses[i];
			let timestampWrites;
			if (slot) {
				const isFirst = i === 0;
				const isLast = i === filteredPasses.length - 1;
				if (isFirst || isLast) timestampWrites = {
					querySet: slot.batch.querySet,
					...isFirst ? { beginningOfPassWriteIndex: slot.beginIndex } : {},
					...isLast ? { endOfPassWriteIndex: slot.endIndex } : {}
				};
			}
			const passEncoder = commandEncoder.beginComputePass({ timestampWrites });
			passEncoder.setPipeline(pipeline);
			passEncoder.setBindGroup(0, bindGroup);
			if (uniformBindGroup) passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
			passEncoder.dispatchWorkgroups(grid[0], grid[1]);
			passEncoder.end();
		}
		if (slot) recordTrace(device, slot, source, filteredPasses.length, shader.code);
	}
	device.queue.submit([commandEncoder.finish()]);
}
function combineUniforms(device, uniforms) {
	for (const buf of uniforms) if (!buf || buf.byteLength === 0 || buf.byteLength !== uniforms[0].byteLength) throw new Error("webgpu: Uniform mismatch between shader passes");
	const minAlign = device.limits.minUniformBufferOffsetAlignment;
	const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
	const buffer = device.createBuffer({
		size: alignment * uniforms.length,
		usage: GPUBufferUsage.UNIFORM,
		mappedAtCreation: true
	});
	const bufferMapped = new Uint8Array(buffer.getMappedRange());
	for (let i = 0; i < uniforms.length; i++) bufferMapped.set(uniforms[i], i * alignment);
	buffer.unmap();
	return [buffer, alignment];
}
/**
* A cache for compiled GPU compute pipelines, keyed by the shader source.
*
* This supports both async compilation (recommended) and a synchronous variant.
* If the pipeline is not in the cache, it will be compiled and added. For async
* compilation, only one compilation will be in progress at a time for a given
* shader source.
*/
var ShaderPipelineCache = class {
	device;
	cache;
	inProgress;
	constructor(device) {
		this.device = device;
		this.cache = /* @__PURE__ */ new Map();
		this.inProgress = /* @__PURE__ */ new Map();
	}
	#getLayout(shader) {
		if (shader.numInputs + shader.numOutputs > this.device.limits.maxStorageBuffersPerShaderStage) {
			const actual = shader.numInputs + shader.numOutputs;
			const max = this.device.limits.maxStorageBuffersPerShaderStage;
			throw new Error(`Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`);
		}
		const bindGroupLayouts = [this.device.createBindGroupLayout({ entries: require_backend.range(shader.numInputs + shader.numOutputs).map((i) => ({
			binding: i,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: i < shader.numInputs ? "read-only-storage" : "storage" }
		})) })];
		if (shader.hasUniform) bindGroupLayouts.push(this.device.createBindGroupLayout({ entries: [{
			binding: 0,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {
				type: "uniform",
				hasDynamicOffset: true
			}
		}] }));
		return this.device.createPipelineLayout({ bindGroupLayouts });
	}
	async prepare(shader) {
		const existingPipeline = this.cache.get(shader.code);
		if (existingPipeline) return existingPipeline;
		const existingPromise = this.inProgress.get(shader.code);
		if (existingPromise) return await existingPromise;
		if (require_backend.DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
		const shaderModule = this.device.createShaderModule({ code: shader.code });
		const promise = (async () => {
			this.device.pushErrorScope("validation");
			try {
				const pipeline = await this.device.createComputePipelineAsync({
					layout: this.#getLayout(shader),
					compute: {
						module: shaderModule,
						entryPoint: "main"
					}
				});
				await this.device.popErrorScope();
				return pipeline;
			} catch (_error) {
				const emsg = await compileError(shaderModule, await this.device.popErrorScope(), shader.code);
				throw new Error(emsg);
			}
		})();
		this.inProgress.set(shader.code, promise);
		const pipeline = await promise;
		this.cache.set(shader.code, pipeline);
		return pipeline;
	}
	prepareSync(shader) {
		const existingPipeline = this.cache.get(shader.code);
		if (existingPipeline) return existingPipeline;
		if (require_backend.DEBUG >= 2) console.info("=========== WebGPU shader ===========\n" + shader.code);
		const shaderModule = this.device.createShaderModule({ code: shader.code });
		this.device.pushErrorScope("validation");
		const pipeline = this.device.createComputePipeline({
			layout: this.#getLayout(shader),
			compute: {
				module: shaderModule,
				entryPoint: "main"
			}
		});
		this.device.popErrorScope().then(async (scope) => {
			if (scope !== null) {
				const emsg = await compileError(shaderModule, scope, shader.code);
				console.error(emsg);
			}
		});
		this.cache.set(shader.code, pipeline);
		return pipeline;
	}
};
/** Gather information about a compilation error and format it. */
async function compileError(shaderModule, scope, code) {
	let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
	const info = await shaderModule.getCompilationInfo();
	for (const msg of info.messages) message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
	if (code) message += `\n\n${code}`;
	return message;
}
//#endregion
exports.WebGPUBackend = WebGPUBackend;
