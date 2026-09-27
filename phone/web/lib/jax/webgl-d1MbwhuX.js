import { $ as range, F as isFloatDtype, L as DEBUG, T as AluGroup, g as tuneNullopt, i as UnsupportedRoutineError, it as strip1, n as SlotError, r as UnsupportedOpError, t as Executable } from "./backend-CnjgxpXq.js";
//#region src/backend/webgl/builtins.ts
const threefrySrc = `
uvec2 threefry2x32(uvec2 key, uvec2 ctr) {
  uint ks0 = key.x;
  uint ks1 = key.y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  uint x0 = ctr.x + ks0;
  uint x1 = ctr.y + ks1;

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

  return uvec2(x0, x1);
}`;
const erfSrc = `
const float _erf_p = 0.3275911;
const float _erf_a1 = 0.254829592;
const float _erf_a2 = -0.284496736;
const float _erf_a3 = 1.421413741;
const float _erf_a4 = -1.453152027;
const float _erf_a5 = 1.061405429;
float erf(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
float erfc(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  float E = P_t * exp(-x * x);
  return x >= 0.0 ? E : 2.0 - E;
}`;
const bitCountSrc = `
int bitCountFallback(uint value) {
  value = value - ((value >> 1u) & 0x55555555u);
  value = (value & 0x33333333u) + ((value >> 2u) & 0x33333333u);
  value = (value + (value >> 4u)) & 0x0f0f0f0fu;
  value = value + (value >> 8u);
  value = value + (value >> 16u);
  return int(value & 0x3fu);
}
int bitCountFallback(int value) {
  return bitCountFallback(uint(value));
}`;
//#endregion
//#region src/backend/webgl.ts
/**
* No-frills backend that uses WebGL2 textures and shaders for compute.
*
* WebGL2 is available in almost all modern browsers, and it has options for
* floating-point numbers and integers in textures. However, it's still not a
* "real" compute API, and only float32 arithmetic is available.
*
* We make this backend available in case users want a fallback option compared
* to WebGPU, which is only on newer browsers and iOS 26+.
*
* Implementation notes:
* - All data is stored in typed RGBA32F textures, regardless of original dtype.
*   They are converted to the correct data type when loaded in shaders.
* - Each texel holds 4 float32 values (128 bits).
* - Compute is done by rendering a full-screen quad with a fragment shader.
* - Output is rendered to a framebuffer-attached texture, then read back.
*/
var WebGLBackend = class {
	type = "webgl";
	maxArgs = 8;
	gl;
	#fbo;
	#buffers;
	#programCache;
	#nextSlot;
	constructor(gl) {
		this.gl = gl;
		this.#fbo = gl.createFramebuffer();
		this.#buffers = /* @__PURE__ */ new Map();
		this.#programCache = /* @__PURE__ */ new Map();
		this.#nextSlot = 1;
	}
	/**
	* Allocate a buffer with a specific dtype.
	*
	* All buffers use RGBA32F texture format internally. Data is stored as raw
	* bits and reinterpreted using floatBitsToInt/intBitsToFloat in shaders.
	* This mirrors how WebGPU handles untyped byte buffers.
	*/
	malloc(size, initialData) {
		const gl = this.gl;
		const numFloats = Math.ceil(size / 4) || 1;
		const { width, height } = computeTextureDimensions(Math.ceil(numFloats / 4) || 1);
		const texture = gl.createTexture();
		if (!texture) throw new Error("Failed to create texture");
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		const totalFloats = width * height * 4;
		let pixels = null;
		if (initialData) {
			pixels = new Float32Array(totalFloats);
			new Uint8Array(pixels.buffer).set(initialData);
		}
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, pixels);
		gl.bindTexture(gl.TEXTURE_2D, null);
		const slot = this.#nextSlot++;
		this.#buffers.set(slot, {
			ref: 1,
			size,
			texture,
			width,
			height
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
			this.gl.deleteTexture(buffer.texture);
			this.#buffers.delete(slot);
		}
	}
	async read(slot, start, count) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		const gl = this.gl;
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.size - start;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
		const totalBytes = buffer.width * buffer.height * 4 * 4;
		const floatData = new Float32Array(totalBytes / 4);
		const pbo = gl.createBuffer();
		if (!pbo) throw new Error("Failed to create PBO");
		gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
		gl.bufferData(gl.PIXEL_PACK_BUFFER, totalBytes, gl.STREAM_READ);
		gl.readPixels(0, 0, buffer.width, buffer.height, gl.RGBA, gl.FLOAT, 0);
		const readError = gl.getError();
		if (readError !== gl.NO_ERROR) {
			gl.deleteBuffer(pbo);
			throw new Error(`WebGL error after readPixels: ${readError}`);
		}
		const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
		if (!sync) throw new Error("Failed to create sync object");
		gl.flush();
		gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		await new Promise((resolve, reject) => {
			const poll = () => {
				const status = gl.clientWaitSync(sync, 0, 0);
				if (status === gl.TIMEOUT_EXPIRED) {
					setTimeout(poll, 5);
					return;
				}
				if (status === gl.WAIT_FAILED) {
					gl.deleteSync(sync);
					gl.deleteBuffer(pbo);
					reject(/* @__PURE__ */ new Error("clientWaitSync failed"));
					return;
				}
				resolve();
			};
			poll();
		});
		gl.deleteSync(sync);
		gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
		gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, floatData);
		gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
		gl.deleteBuffer(pbo);
		const byteData = new Uint8Array(floatData.buffer);
		return new Uint8Array(byteData.slice(start, start + count));
	}
	readSync(slot, start, count) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		const gl = this.gl;
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.size - start;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
		const totalFloats = buffer.width * buffer.height * 4;
		const floatData = new Float32Array(totalFloats);
		gl.readPixels(0, 0, buffer.width, buffer.height, gl.RGBA, gl.FLOAT, floatData);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		const byteData = new Uint8Array(floatData.buffer);
		return new Uint8Array(byteData.slice(start, start + count));
	}
	async prepareKernel(kernel) {
		return this.prepareKernelSync(kernel);
	}
	prepareKernelSync(kernel) {
		const shader = generateShader(kernel);
		const cached = this.#programCache.get(shader.code);
		if (cached) return new Executable(kernel, cached);
		const dispatch = compileShader(this.gl, shader);
		this.#programCache.set(shader.code, dispatch);
		return new Executable(kernel, dispatch);
	}
	prepareRoutine(routine) {
		throw new UnsupportedRoutineError(routine.name, "webgl");
	}
	prepareRoutineSync(routine) {
		throw new UnsupportedRoutineError(routine.name, "webgl");
	}
	dispatch(exe, inputs, outputs) {
		const gl = this.gl;
		if (gl.isContextLost()) throw new Error("WebGL context lost - cannot dispatch");
		const { program, inputLocations } = exe.data;
		if (inputs.length !== exe.data.numInputs) throw new Error(`Expected ${exe.data.numInputs} inputs, got ${inputs.length}`);
		if (outputs.length !== 1) throw new Error(`Expected 1 output, got ${outputs.length}`);
		const outputBuffer = this.#buffers.get(outputs[0]);
		if (!outputBuffer) throw new SlotError(outputs[0]);
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputBuffer.texture, 0);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Framebuffer incomplete: ${status}`);
		gl.viewport(0, 0, outputBuffer.width, outputBuffer.height);
		gl.useProgram(program);
		for (let i = 0; i < inputs.length; i++) {
			const inputBuffer = this.#buffers.get(inputs[i]);
			if (!inputBuffer) throw new SlotError(inputs[i]);
			gl.activeTexture(gl.TEXTURE0 + i);
			gl.bindTexture(gl.TEXTURE_2D, inputBuffer.texture);
			if (inputLocations[i] !== null) gl.uniform1i(inputLocations[i], i);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		const error = gl.getError();
		if (error !== gl.NO_ERROR) {
			let errorName;
			if (error === gl.INVALID_ENUM) errorName = "INVALID_ENUM";
			else if (error === gl.INVALID_VALUE) errorName = "INVALID_VALUE";
			else if (error === gl.INVALID_OPERATION) errorName = "INVALID_OPERATION";
			else if (error === gl.INVALID_FRAMEBUFFER_OPERATION) errorName = "INVALID_FRAMEBUFFER_OPERATION";
			else if (error === gl.OUT_OF_MEMORY) errorName = "OUT_OF_MEMORY";
			else if (error === gl.CONTEXT_LOST_WEBGL) errorName = "CONTEXT_LOST_WEBGL";
			else errorName = `UNKNOWN(${error})`;
			throw new Error(`WebGL error after drawArrays: ${errorName}`);
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.useProgram(null);
	}
};
function generateShader(kernel) {
	const tune = tuneNullopt(kernel);
	if (DEBUG >= 3) console.info(`webgl kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
	const { nargs, reduction: re } = kernel;
	const outputDtype = kernel.dtype;
	const outputSize = computeTextureDimensions(Math.ceil(kernel.size / 4) || 1);
	const inputDtypes = Array(nargs).fill("float32");
	const builtins = {
		bitCount: false,
		erf: false,
		threefry: false
	};
	const collectInfo = (exp) => {
		if (exp.op === "GlobalIndex") inputDtypes[exp.arg[0]] = exp.dtype;
		else if (exp.op === "Erf" || exp.op === "Erfc") builtins.erf = true;
		else if (exp.op === "BitCount") builtins.bitCount = true;
		else if (exp.op === "Threefry2x32") builtins.threefry = true;
	};
	tune.exp.fold(collectInfo);
	tune.epilogue?.fold(collectInfo);
	const shader = [];
	let indent = "";
	const pushIndent = Symbol("pushIndent");
	const popIndent = Symbol("popIndent");
	const emit = (...lines) => {
		for (const line of lines) if (line === pushIndent) indent += "  ";
		else if (line === popIndent) indent = indent.slice(0, -2);
		else shader.push(line ? indent + line : line);
	};
	emit("#version 300 es", "precision highp float;", "precision highp int;", "");
	const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
	const resultType = glslType(outputDtype);
	for (let i = 0; i < nargs; i++) emit(`uniform highp sampler2D ${args[i]};`);
	emit("out vec4 out0;");
	const fetchFunctions = /* @__PURE__ */ new Set();
	for (const dtype of inputDtypes) fetchFunctions.add(dtype);
	for (const dtype of fetchFunctions) emit(generateLoadFunction(dtype));
	if (builtins.erf) emit(erfSrc);
	if (builtins.bitCount) emit(bitCountSrc);
	if (builtins.threefry) emit(threefrySrc);
	emit(`${resultType} compute(int gidx) {`, pushIndent, `${resultType} result = ${constToGlsl(outputDtype, 0)};`, `if (gidx < ${kernel.size}) {`, pushIndent);
	if (!re) emit(`result = ${strip1(generateExpression(tune.exp, args, inputDtypes))};`);
	else {
		emit(`${glslType(re.dtype)} acc = ${constToGlsl(re.dtype, re.identity)};`, `for (int ridx = 0; ridx < ${tune.size.reduce}; ridx++) {`, pushIndent);
		const code = generateExpression(tune.exp, args, inputDtypes);
		if (re.op === "Add") emit(`acc += ${strip1(code)};`);
		else if (re.op === "Mul") emit(`acc *= ${strip1(code)};`);
		else if (re.op === "Min") if (re.dtype !== "bool") emit(`acc = min(acc, ${strip1(code)});`);
		else emit(`acc = acc && ${code};`);
		else if (re.op === "Max") if (re.dtype !== "bool") emit(`acc = max(acc, ${strip1(code)});`);
		else emit(`acc = acc || ${code};`);
		else throw new Error(`Unsupported reduction op: ${re.op}`);
		emit(popIndent, "}");
		emit(`result = ${generateExpression(tune.epilogue, args, inputDtypes)};`);
	}
	emit(popIndent, "}", "return result;", popIndent, "}\n");
	emit("void main() {", pushIndent, "ivec2 fragCoord = ivec2(gl_FragCoord.xy);", `int texelIdx = fragCoord.y * ${outputSize.width} + fragCoord.x;`, `${resultType} result0 = compute(texelIdx * 4);`, `${resultType} result1 = compute(texelIdx * 4 + 1);`, `${resultType} result2 = compute(texelIdx * 4 + 2);`, `${resultType} result3 = compute(texelIdx * 4 + 3);`, `out0 = vec4(${range(4).map((i) => toRGBA32F(outputDtype, `result${i}`)).join(", ")});`);
	emit(popIndent, "}");
	return {
		code: shader.join("\n"),
		numInputs: nargs,
		outputSize: [outputSize.width, outputSize.height],
		outputDtype
	};
}
function compile(gl, type, src) {
	const s = gl.createShader(type);
	gl.shaderSource(s, src);
	gl.compileShader(s);
	if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "Unknown shader compile error");
	return s;
}
function link(gl, vsSrc, fsSrc) {
	const p = gl.createProgram();
	gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
	gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
	gl.linkProgram(p);
	if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "Unknown program link error");
	return p;
}
const vertexShaderSource = `#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;
function compileShader(gl, shader) {
	if (DEBUG >= 1) console.info("=========== WebGL shader ===========\n" + shader.code);
	const program = link(gl, vertexShaderSource, shader.code);
	const inputLocations = [];
	for (let i = 0; i < shader.numInputs; i++) inputLocations.push(gl.getUniformLocation(program, `in${i}`));
	return {
		...shader,
		program,
		inputLocations
	};
}
/** Compute 2D texture dimensions for a given number of texels. */
function computeTextureDimensions(numTexels) {
	const maxDim = 16384;
	let width = Math.min(Math.ceil(Math.sqrt(numTexels)), maxDim);
	width = Math.min(1 << Math.ceil(Math.log2(width)), maxDim);
	const height = Math.min(Math.ceil(numTexels / width), maxDim);
	return {
		width,
		height
	};
}
function glslType(dtype) {
	switch (dtype) {
		case "float32": return "float";
		case "int32": return "int";
		case "uint32": return "uint";
		case "bool": return "bool";
		default: throw new Error(`Unsupported dtype for WebGL: ${dtype}`);
	}
}
function generateLoadFunction(dtype) {
	const funcName = `load_${dtype}`;
	const returnType = glslType(dtype);
	let conversion;
	if (isFloatDtype(dtype)) conversion = "val";
	else if (dtype === "int32") conversion = "floatBitsToInt(val)";
	else if (dtype === "uint32") conversion = "floatBitsToUint(val)";
	else if (dtype === "bool") conversion = "floatBitsToInt(val) != 0";
	else throw new Error(`Unsupported dtype for WebGL fetch: ${dtype}`);
	return `
${returnType} ${funcName}(highp sampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  vec4 texVal = texelFetch(tex, coord, 0);
  float val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${conversion};
}
`;
}
function toRGBA32F(dtype, source) {
	switch (dtype) {
		case "float32": return source;
		case "int32": return `intBitsToFloat(${source})`;
		case "uint32": return `uintBitsToFloat(${source})`;
		case "bool": return `intBitsToFloat(${source} ? 1 : 0)`;
		default: throw new Error(`Unsupported dtype for WebGL output: ${dtype}`);
	}
}
function constToGlsl(dtype, value) {
	switch (dtype) {
		case "bool": return value ? "true" : "false";
		case "int32": return value.toString();
		case "uint32": return value.toString() + "u";
		case "float32":
			if (Number.isNaN(value)) return "uintBitsToFloat(0x7fc00000u)";
			if (!Number.isFinite(value)) return value > 0 ? "uintBitsToFloat(0x7f800000u)" : "uintBitsToFloat(0xff800000u)";
			return "float(" + value.toString() + ")";
		default: throw new Error(`Unsupported dtype for WebGL constant: ${dtype}`);
	}
}
/** Generate GLSL expression code from an AluExp. */
function generateExpression(exp, args, inputDtypes) {
	const expContext = /* @__PURE__ */ new Map();
	const gen = (e) => {
		if (expContext.has(e)) return expContext.get(e);
		const { op, src, dtype, arg } = e;
		let source = "";
		if (AluGroup.Binary.has(op)) {
			const a = gen(src[0]);
			const b = gen(src[1]);
			if (op === "Add") if (dtype === "bool") source = `(${a} || ${b})`;
			else source = `(${a} + ${b})`;
			else if (op === "Sub") source = `(${a} - ${b})`;
			else if (op === "Mul") if (dtype === "bool") source = `(${a} && ${b})`;
			else source = `(${a} * ${b})`;
			else if (op === "Idiv") if (isFloatDtype(dtype)) source = `trunc(${a} / ${b})`;
			else source = `(${a} / ${b})`;
			else if (op === "Mod") if (isFloatDtype(dtype)) source = `(${a} - ${b} * trunc(${a} / ${b}))`;
			else source = `(${a} % ${b})`;
			else if (op === "Min") if (dtype === "bool") source = `(${a} && ${b})`;
			else source = `min(${a}, ${b})`;
			else if (op === "Max") if (dtype === "bool") source = `(${a} || ${b})`;
			else source = `max(${a}, ${b})`;
			else if (op === "BitCombine") {
				let infix = arg === "and" ? "&" : arg === "or" ? "|" : "^";
				if (dtype === "bool") infix = infix + infix;
				source = `(${a} ${infix} ${b})`;
			} else if (op === "BitShift") if (arg === "shl") source = `(${a} << ${b})`;
			else source = `(${a} >> ${b})`;
		} else if (AluGroup.Compare.has(op)) {
			const a = gen(src[0]);
			const b = gen(src[1]);
			if (op === "Cmplt") source = `(${a} < ${b})`;
			else if (op === "Cmpne") if (isFloatDtype(src[0].dtype)) source = `(${a} != ${b} || isnan(${a}) || isnan(${b}))`;
			else source = `(${a} != ${b})`;
		} else if (AluGroup.Unary.has(op)) {
			const a = gen(src[0]);
			if (op === "BitCount") source = `bitCountFallback(${strip1(a)})`;
			else if (op === "Sin") source = `sin(${strip1(a)})`;
			else if (op === "Cos") source = `cos(${strip1(a)})`;
			else if (op === "Asin") source = `asin(${strip1(a)})`;
			else if (op === "Atan") source = `atan(${strip1(a)})`;
			else if (op === "Exp") source = `exp(${strip1(a)})`;
			else if (op === "Log") source = `log(${strip1(a)})`;
			else if (op === "Erf") source = `erf(${strip1(a)})`;
			else if (op === "Erfc") source = `erfc(${strip1(a)})`;
			else if (op === "Sqrt") source = `sqrt(${strip1(a)})`;
			else if (op === "Floor") source = `floor(${strip1(a)})`;
			else if (op === "Ceil") source = `ceil(${strip1(a)})`;
			else if (op === "Reciprocal") source = `(1.0 / ${a})`;
			else if (op === "Cast") source = `${glslType(dtype)}(${strip1(a)})`;
			else if (op === "Bitcast") {
				const dtype0 = src[0].dtype;
				if (dtype === dtype0) source = a;
				else if (dtype === "float32") {
					if (dtype0 === "int32") source = `intBitsToFloat(${strip1(a)})`;
					else if (dtype0 === "uint32") source = `uintBitsToFloat(${strip1(a)})`;
				} else if (dtype === "int32") {
					if (dtype0 === "float32") source = `floatBitsToInt(${strip1(a)})`;
					else if (dtype0 === "uint32") source = `int(${strip1(a)})`;
				} else if (dtype === "uint32") {
					if (dtype0 === "float32") source = `floatBitsToUint(${strip1(a)})`;
					else if (dtype0 === "int32") source = `uint(${strip1(a)})`;
				}
			}
		} else if (op === "Threefry2x32") {
			const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
			const mode = arg;
			const call = `threefry2x32(uvec2(${k0}, ${k1}), uvec2(${c0}, ${c1}))`;
			if (mode === "xor") source = `(${call}.x ^ ${call}.y)`;
			else if (mode === 0) source = `${call}.x`;
			else if (mode === 1) source = `${call}.y`;
		} else if (op === "Where") {
			const [cond, t, f] = src.map(gen);
			source = `(${cond} ? ${t} : ${f})`;
		} else if (op === "Const") source = constToGlsl(dtype, arg);
		else if (op === "Special") source = arg[0];
		else if (op === "Variable") source = arg;
		else if (op === "GlobalIndex") {
			const gid = arg[0];
			const bufidx = gen(src[0]);
			source = `load_${inputDtypes[gid]}(${args[gid]}, ${strip1(bufidx)})`;
		}
		if (!source) throw new UnsupportedOpError(op, dtype, "webgl", arg);
		expContext.set(e, source);
		return source;
	};
	return gen(exp);
}
//#endregion
export { WebGLBackend };
