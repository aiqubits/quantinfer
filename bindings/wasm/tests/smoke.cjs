const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const [binding, modelPath, goldenPath, expectedTokens] = process.argv.slice(2);
const { Model } = require(path.resolve(binding));
const golden = JSON.parse(fs.readFileSync(goldenPath, "utf8"));
const expected = JSON.parse(expectedTokens);

assert.throws(() => new Model(new Uint8Array(4)), /GGUF|truncated/);
const model = new Model(fs.readFileSync(modelPath));
try {
    const prompt = model.tokenize(golden.prompt);
    assert.ok(prompt instanceof Uint32Array);
    assert.deepEqual([...prompt], golden.prompt_tokens);
    const tokens = model.generateTokens(prompt, expected.length);
    assert.ok(tokens instanceof Uint32Array);
    assert.deepEqual([...tokens], expected);
    assert.equal(model.generate(golden.prompt, expected.length), model.detokenize(tokens));
    assert.throws(() => model.generateTokens(new Uint32Array([0xffffffff]), 1), /valid token IDs/);
    assert.throws(() => model.generateTokens(prompt, 0), /max_new_tokens/);
} finally {
    model.free();
}
