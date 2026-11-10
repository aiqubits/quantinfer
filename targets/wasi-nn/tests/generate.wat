(module
  (import "wasi_ephemeral_nn" "load_by_name" (func $load (param i32 i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "init_execution_context" (func $init (param i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "set_input" (func $set (param i32 i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "compute" (func $compute (param i32) (result i32)))
  (import "wasi_ephemeral_nn" "get_output" (func $get (param i32 i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)

  (data (i32.const 0) "smol")
  ;; Tensor dimensions [5] and [1].
  (data (i32.const 8) "\05\00\00\00\01\00\00\00")
  ;; Prompt token IDs followed by max_new_tokens.
  (data (i32.const 16) "\f8\01\00\00\f7\0d\00\00\1a\01\00\00\29\12\00\00\3a\01\00\00\08\00\00\00")
  ;; WITX tensor records. Dimensions pointer/count, I32 tag, data pointer/count.
  (data (i32.const 48) "\08\00\00\00\01\00\00\00\04\00\00\00\10\00\00\00\14\00\00\00")
  (data (i32.const 68) "\0c\00\00\00\01\00\00\00\04\00\00\00\24\00\00\00\04\00\00\00")

  (func $check (param $errno i32)
    (if (local.get $errno) (then unreachable)))

  (func (export "generate") (result i32)
    (local $context i32)
    (call $check (call $load (i32.const 0) (i32.const 4) (i32.const 96)))
    (call $check (call $init (i32.load (i32.const 96)) (i32.const 100)))
    (local.set $context (i32.load (i32.const 100)))
    (call $check (call $set (local.get $context) (i32.const 0) (i32.const 48)))
    (call $check (call $set (local.get $context) (i32.const 1) (i32.const 68)))
    (call $check (call $compute (local.get $context)))
    (call $check (call $get (local.get $context) (i32.const 0) (i32.const 128) (i32.const 32) (i32.const 104)))
    (i32.load (i32.const 104)))
)
