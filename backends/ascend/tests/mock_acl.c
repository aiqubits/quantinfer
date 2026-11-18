/* Independent CANN ABI contract double. It defers matmul until stream sync,
 * counts ownership, and injects failures. It is not an NPU implementation. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int64_t rows, columns; float *data; } Tensor;
typedef struct { Tensor *a, *b, *c; int reusable; } Plan;
static int phase, nth, allocation_calls, tensor_calls;
static int initialized, finalized, allocations, tensors, contexts, streams, plans;
static int queries, executions, copies, synchronizations, violations;
static int external_context;
static _Thread_local void *current;
static Plan *pending;

void mock_configure(int p, int n) { phase=p; nth=n; allocation_calls=0; tensor_calls=0; }
void mock_external_context(void) { current=&external_context; }
int mock_stat(int i) {
    int stats[]={initialized,finalized,allocations,tensors,contexts,streams,plans,
        queries,executions,copies,synchronizations,violations,current==&external_context};
    return i>=0 && (size_t)i<sizeof(stats)/sizeof(stats[0]) ? stats[i] : -1;
}
int aclInit(const char *config) { (void)config; ++initialized; return 0; }
int aclFinalize(void) { ++finalized; return 0; }
int aclrtGetDeviceCount(uint32_t *count) { *count=1; return phase==9 ? 77 : 0; }
int aclrtGetCurrentContext(void **context) { *context=current; return current ? 0 : 107002; }
int aclrtCreateContext(void **context, int device) {
    if (device!=0) return 77;
    if (phase==7) { *context=(void *)(uintptr_t)1; return 77; }
    *context=calloc(1,16); if (!*context) return 77;
    ++contexts; current=*context; return 0;
}
int aclrtSetCurrentContext(void *context) {
    if (phase==13 && context==&external_context) return 77;
    current=context; return 0;
}
int aclrtDestroyContext(void *context) {
    if (context==(void *)(uintptr_t)1) { ++violations; return 77; }
    if (pending) { ++violations; return 77; }
    if (current==context) current=NULL;
    free(context); --contexts; return 0;
}
int aclrtCreateStream(void **stream) {
    if (!current) return 77;
    if (phase==8) { *stream=(void *)(uintptr_t)1; return 77; }
    *stream=calloc(1,16); if (!*stream) return 77; ++streams; return 0;
}
int aclrtDestroyStream(void *stream) {
    if (stream==(void *)(uintptr_t)1 || pending) { ++violations; return 77; }
    free(stream); --streams; return 0;
}
int aclrtMalloc(void **ptr, size_t size, int policy) {
    ++allocation_calls;
    if (policy!=0 || !size || !current) { ++violations; return 77; }
    if (phase==1 && allocation_calls==nth) return 77;
    *ptr=calloc(1,size); if (!*ptr) return 77; ++allocations; return 0;
}
int aclrtFree(void *ptr) {
    if (phase==11) return 77;
    if (pending) { ++violations; return 77; }
    free(ptr); --allocations; return 0;
}
int aclrtMemcpy(void *dst, size_t capacity, const void *src, size_t size, int kind) {
    if (!dst || !src || capacity<size || (kind!=1 && kind!=2)) { ++violations; return 77; }
    if (kind==2 && (phase==6 || pending)) return 77;
    ++copies; memcpy(dst,src,size); return 0;
}
void *aclCreateTensor(const int64_t *shape, uint64_t rank, int dtype,
        const int64_t *strides, int64_t offset, int format,
        const int64_t *storage, uint64_t storage_rank, void *data) {
    ++tensor_calls;
    if (phase==2 && tensor_calls==nth) return NULL;
    if (rank!=2 || storage_rank!=2 || dtype!=0 || format!=2 || offset!=0 ||
        strides[0]!=shape[1] || strides[1]!=1 || storage[0]!=shape[0] ||
        storage[1]!=shape[1] || shape[0]<=0 || shape[1]<=0 || !data) {
        ++violations; return NULL;
    }
    Tensor *tensor=malloc(sizeof(*tensor)); if (!tensor) return NULL;
    *tensor=(Tensor){shape[0],shape[1],data}; ++tensors; return tensor;
}
int aclDestroyTensor(void *tensor) {
    if (pending) { ++violations; return 77; }
    free(tensor); --tensors; return 0;
}
int aclnnMmGetWorkspaceSize(const void *a, const void *b, void *c, int8_t math,
        uint64_t *size, void **executor) {
    ++queries;
    if (phase==3) return 77;
    if (!a || !b || !c || math!=0) { ++violations; return 77; }
    Plan *plan=malloc(sizeof(*plan)); if (!plan) return 77;
    *plan=(Plan){(Tensor *)a,(Tensor *)b,c,0}; ++plans;
    *size=32; *executor=plan; return 0;
}
int aclSetAclOpExecutorRepeatable(void *executor) {
    if (phase==10) return 77;
    ((Plan *)executor)->reusable=1; return 0;
}
int aclDestroyAclOpExecutor(void *executor) {
    if (phase==12) return 77;
    if (pending || !((Plan *)executor)->reusable) { ++violations; return 77; }
    free(executor); --plans; return 0;
}
int aclnnMm(void *workspace, uint64_t size, void *executor, void *stream) {
    ++executions;
    if (!workspace || size!=32 || !stream || pending) { ++violations; return 77; }
    pending=executor; return phase==4 ? 77 : 0;
}
int aclrtSynchronizeStream(void *stream) {
    ++synchronizations;
    if (!stream || phase==5) return 77;
    if (pending) {
        Tensor *a=pending->a, *b=pending->b, *c=pending->c;
        if (a->columns!=b->rows || c->rows!=a->rows || c->columns!=b->columns) {
            ++violations; return 77;
        }
        for (int64_t i=0;i<c->rows;++i) for (int64_t j=0;j<c->columns;++j) {
            float sum=0;
            for (int64_t k=0;k<a->columns;++k) sum+=a->data[i*a->columns+k]*b->data[k*b->columns+j];
            c->data[i*c->columns+j]=sum;
        }
        pending=NULL;
    }
    return 0;
}
