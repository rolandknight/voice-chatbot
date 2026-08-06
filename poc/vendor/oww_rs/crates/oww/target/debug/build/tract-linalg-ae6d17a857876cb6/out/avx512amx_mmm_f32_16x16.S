// vim: set syntax=asm :
//
// Intel AMX bf16 GEMM kernel, 16 M-rows x 16 N-cols f32 accumulator output.
//
// One `tdpbf16ps tmm0, tmm1, tmm2` instruction performs:
//   tmm0[m, n] += sum_{k=0..31} A[m, k] * B[k, n]   (multiplies in bf16,
//                                                    accumulates in f32)
// for m=0..15, n=0..15: 16 * 16 * 32 = 8192 fma per single instruction --
// the same throughput as TDPBSSD on the same hardware. Accelerates f32
// matmul on Sapphire Rapids+ at the cost of bf16 input truncation
// (~1/256 relative error per multiply; sqrt(K) compounded by FMA chain).
//
// Tile geometry (palette 1, the maximum-bytes AMX tile shape):
//   tmm0 = C accumulator: 16 rows x 64 colsb = 16 M-rows x 16 N-cols of f32
//   tmm1 = A tile:        16 rows x 64 colsb = 16 M-rows x 32 K-bf16 / iter
//   tmm2 = B tile:        16 rows x 64 colsb = 16 K-pair-rows x 16 N x 2 bf16
//
// A is packed via PackedAmxBf16A(16): per panel of 16 M-rows, row-major
// within the panel, K-bf16 contiguous along the row, K_padded =
// ceil(K/32)*32 bf16. Source f32 is truncated to bf16 at pack time using
// round-to-nearest-even (matches VCVTNEPS2BF16 semantics).
//
// B is packed via PackedBf16K2(16): per K=2 block, 16 N-cols x 2 K-bf16 =
// 64 bytes; 16 K-blocks per tmm2 tile. Source f32 -> bf16 same as A.
//
// REGISTER LAYOUT (mirrors the i32 16x16 sibling):
//   zmm0..zmm15 = accumulators, ROW-MAJOR: zmm{m} = row m of C as 16 f32
//                 lanes [C[m, 0], C[m, 1], ..., C[m, 15]].



.intel_syntax noprefix
.text
.p2align 5
.globl avx512amx_mmm_f32_16x16_0_23_4
avx512amx_mmm_f32_16x16_0_23_4:
.cfi_startproc



    push        rbp
    mov         rbp, rsp



    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    sub         rsp, 8


.cfi_def_cfa_offset 64


    stmxcsr     [rsp + 4]

    mov         rax, 0x1FC0

    mov         [rsp], eax
    ldmxcsr     [rsp]

    // Reserve 64 bytes for the AMX tile-config block, zero it, populate
    // palette + dims (all three tiles are 16 rows x 64 colsb). Same shape
    // as the i32 16x16 sibling.
    sub         rsp, 64
    vpxor       xmm15, xmm15, xmm15
    vmovdqu     [rsp     ], xmm15
    vmovdqu     [rsp + 16], xmm15
    vmovdqu     [rsp + 32], xmm15
    vmovdqu     [rsp + 48], xmm15
    mov         byte ptr  [rsp + 0 ], 1    // palette = 1
    mov         word ptr  [rsp + 16], 64   // colsb[0] = 64 (tmm0)
    mov         word ptr  [rsp + 18], 64   // colsb[1] = 64 (tmm1)
    mov         word ptr  [rsp + 20], 64   // colsb[2] = 64 (tmm2)
    mov         byte ptr  [rsp + 48], 16   // rows[0]  = 16 (tmm0)
    mov         byte ptr  [rsp + 49], 16   // rows[1]  = 16 (tmm1)
    mov         byte ptr  [rsp + 50], 16   // rows[2]  = 16 (tmm2)
    ldtilecfg   [rsp]

// vim: set syntax=asm :

.Lnon_linear:

.Lnon_linear_loop_enter:
    sub     rdi,    40
.Lnon_linear_loop:
    add     rdi,    40
    mov     rax,    [rdi]

    mov     r8, 29
    cmp     rax, 0
    cmovl   rax, r8
    cmp     rax, 29
    cmovg   rax, r8


    lea     r8, [ rip + .Ljmp_table ]

    movsxd  r9, dword ptr [ r8 + rax * 4 ]
    lea     r8, [ r8 + r9 ]
    jmp     r8

.Ljmp_table:

    .long      .Ldone-.Ljmp_table

    .long      .Lclear-.Ljmp_table

    .long      .Lload_tile-.Ljmp_table

    .long      .Lscalar_min-.Ljmp_table

    .long      .Lscalar_max-.Ljmp_table

    .long      .Lscalar_add-.Ljmp_table

    .long      .Lscalar_mul-.Ljmp_table

    .long      .Lscalar_sub-.Ljmp_table

    .long      .Lscalar_sub_flipped-.Ljmp_table

    .long      .Lleaky_relu-.Ljmp_table

    .long      .Lper_row_min-.Ljmp_table

    .long      .Lper_row_max-.Ljmp_table

    .long      .Lper_row_add-.Ljmp_table

    .long      .Lper_row_mul-.Ljmp_table

    .long      .Lper_row_sub-.Ljmp_table

    .long      .Lper_row_sub_flipped-.Ljmp_table

    .long      .Lper_col_min-.Ljmp_table

    .long      .Lper_col_max-.Ljmp_table

    .long      .Lper_col_add-.Ljmp_table

    .long      .Lper_col_mul-.Ljmp_table

    .long      .Lper_col_sub-.Ljmp_table

    .long      .Lper_col_sub_flipped-.Ljmp_table

    .long      .Lq_scale-.Ljmp_table

    .long      .Lq_shr-.Ljmp_table

    .long      .Lq_shl-.Ljmp_table

    .long      .Ladd_unicast-.Ljmp_table

    .long      .Ladd_row_col_products-.Ljmp_table

    .long      .Lstore-.Ljmp_table

    .long      .Ladd_mat_mul-.Ljmp_table

    .long      .Lunsupported-.Ljmp_table

.Lunsupported:
    mov     rax,    1
    jmp     .Lreturn


.Ldone:
    mov     rax, 0
    jmp     .Lreturn


.Lclear:
    
    vpxorq      zmm0, zmm0, zmm0
    
    vpxorq      zmm1, zmm1, zmm1
    
    vpxorq      zmm2, zmm2, zmm2
    
    vpxorq      zmm3, zmm3, zmm3
    
    vpxorq      zmm4, zmm4, zmm4
    
    vpxorq      zmm5, zmm5, zmm5
    
    vpxorq      zmm6, zmm6, zmm6
    
    vpxorq      zmm7, zmm7, zmm7
    
    vpxorq      zmm8, zmm8, zmm8
    
    vpxorq      zmm9, zmm9, zmm9
    
    vpxorq      zmm10, zmm10, zmm10
    
    vpxorq      zmm11, zmm11, zmm11
    
    vpxorq      zmm12, zmm12, zmm12
    
    vpxorq      zmm13, zmm13, zmm13
    
    vpxorq      zmm14, zmm14, zmm14
    
    vpxorq      zmm15, zmm15, zmm15
    
    jmp     .Lnon_linear_loop

.Ladd_mat_mul:
    mov     r12,    [rdi + 32]   // packing
    mov     rbx,    [rdi + 24]   // B
    mov     rax,    [rdi + 16]   // A

    mov     rcx,    [rdi + 8]    // k
    test    rcx,    rcx
    jz      .Lnon_linear_loop

    cmp     r12, 1
    je      .Lmain_loop_packed_packed_bf16

.Lmain_loop_packed_packed:
    // Generic f32 x f32 fallback path (non-AMX). For row-major
    // accumulators zmm{m} = row m of C, accumulating C[m, n] += A[m, k] *
    // B[k, n]: load 16 B values for this K row into zmm16, then for each
    // m broadcast A[m, k] and FMA add to zmm{m}.
    vmovups         zmm16, [rbx]               // 16 f32 of B at this K row

    
        vbroadcastss    zmm17, dword ptr [rax + 0 * 4]
        vfmadd231ps     zmm0, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 1 * 4]
        vfmadd231ps     zmm1, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 2 * 4]
        vfmadd231ps     zmm2, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 3 * 4]
        vfmadd231ps     zmm3, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 4 * 4]
        vfmadd231ps     zmm4, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 5 * 4]
        vfmadd231ps     zmm5, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 6 * 4]
        vfmadd231ps     zmm6, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 7 * 4]
        vfmadd231ps     zmm7, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 8 * 4]
        vfmadd231ps     zmm8, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 9 * 4]
        vfmadd231ps     zmm9, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 10 * 4]
        vfmadd231ps     zmm10, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 11 * 4]
        vfmadd231ps     zmm11, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 12 * 4]
        vfmadd231ps     zmm12, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 13 * 4]
        vfmadd231ps     zmm13, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 14 * 4]
        vfmadd231ps     zmm14, zmm16, zmm17
    
        vbroadcastss    zmm17, dword ptr [rax + 15 * 4]
        vfmadd231ps     zmm15, zmm16, zmm17
    

    add             rax,    64                 // 16 f32 lanes per K step
    add             rbx,    64
    dec             rcx
    jnz             .Lmain_loop_packed_packed

    jmp             .Lnon_linear_loop

.Lmain_loop_packed_packed_bf16:
    // AMX bf16 layout:
    //   A panel: 16 M-rows x K_padded bf16 ROW-major within the panel
    //            (PackedAmxBf16A, K_padded = ceil(K/32)*32 bf16 =
    //            ceil(K/32)*64 bytes per row).
    //   B panel: PackedBf16K2(16) -- 16 N-cols x 2 K-bf16 per K=2 block,
    //            with 16 K-blocks per tdpbf16ps iter (16 K-pair-rows x
    //            64 colsb).
    //
    // Per tdpbf16ps: tmm0[m, n] += sum_{k=0..31} A[m, k] * B[k, n]
    // with multiplies in bf16 and accumulation in f32. Inner loop steps
    // along K in 32-bf16 chunks (= 64 bytes per A row).

    // r8 <- K_padded_in_bytes = ceil(k/32)*64 = byte-stride between A's
    // M-rows. (Each bf16 is 2 bytes, so K_padded_bf16 * 2.)
    mov             r8,     rcx
    add             r8,     31
    and             r8,     -32
    shl             r8,     1                  // *2 (bf16 = 2 bytes)

    // rcx <- ceil(k/32) = number of K=32 AMX inner iterations.
    add             rcx,    31
    shr             rcx,    5

    // r9 <- 64 = byte-stride between B's K-pair-rows (16 N-cols * 4 bytes
    // per K-pair = 16 * 4 = 64).
    mov             r9,     64

    tilezero        tmm0

.Lloop_32k_amx_bf16_16x16:
    // oneDNN-aligned cache strategy (same as the i32 sibling):
    //   A -> cached (tileloadd + prefetcht0 to L1), reused across N-tiles.
    //   B -> non-temporal (tileloaddt1 + prefetcht1 to L2), streams once.
    // Each iter advances A by 64 bytes and B by 1024 bytes; we prime the
    // first 6 of next-iter's 16 B cache lines and let the SPR HW stream
    // prefetcher cover the remaining 10.
    prefetcht0      [rax + 64]
    prefetcht1      [rbx + 1024]
    prefetcht1      [rbx + 1088]
    prefetcht1      [rbx + 1152]
    prefetcht1      [rbx + 1216]
    prefetcht1      [rbx + 1280]
    prefetcht1      [rbx + 1344]
    tileloadd       tmm1,   [rax + r8 * 1]     // A tile (cached): stride = K_padded_bytes
    tileloaddt1     tmm2,   [rbx + r9 * 1]     // B tile (non-temporal): stride = 64
    tdpbf16ps       tmm0,   tmm1,   tmm2
    add             rax,    64                 // +32 bf16 in A row 0
    add             rbx,    1024               // 16 K-pairs * 64 = 1024 B
    dec             rcx
    jnz             .Lloop_32k_amx_bf16_16x16

    // tmm0 -> stack scratch (16 rows x 64 bytes = 1024 B row-major f32).
    // Each row's 16 f32 are contiguous, so one 64-byte load per row.
    sub             rsp,    1024
    mov             r10,    rsp
    mov             r11,    64
    tilestored      [r10 + r11 * 1], tmm0

    
    vmovups         zmm0, [r10 + 0]
    
    vmovups         zmm1, [r10 + 64]
    
    vmovups         zmm2, [r10 + 128]
    
    vmovups         zmm3, [r10 + 192]
    
    vmovups         zmm4, [r10 + 256]
    
    vmovups         zmm5, [r10 + 320]
    
    vmovups         zmm6, [r10 + 384]
    
    vmovups         zmm7, [r10 + 448]
    
    vmovups         zmm8, [r10 + 512]
    
    vmovups         zmm9, [r10 + 576]
    
    vmovups         zmm10, [r10 + 640]
    
    vmovups         zmm11, [r10 + 704]
    
    vmovups         zmm12, [r10 + 768]
    
    vmovups         zmm13, [r10 + 832]
    
    vmovups         zmm14, [r10 + 896]
    
    vmovups         zmm15, [r10 + 960]
    

    add             rsp,    1024

    jmp             .Lnon_linear_loop

// ---- Scalar / per-row / per-col f32 epilogues ----------------------------

.Lscalar_min:
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vminps zmm0, zmm0, zmm16
    vminps zmm1, zmm1, zmm16
    vminps zmm2, zmm2, zmm16
    vminps zmm3, zmm3, zmm16
    vminps zmm4, zmm4, zmm16
    vminps zmm5, zmm5, zmm16
    vminps zmm6, zmm6, zmm16
    vminps zmm7, zmm7, zmm16
    vminps zmm8, zmm8, zmm16
    vminps zmm9, zmm9, zmm16
    vminps zmm10, zmm10, zmm16
    vminps zmm11, zmm11, zmm16
    vminps zmm12, zmm12, zmm16
    vminps zmm13, zmm13, zmm16
    vminps zmm14, zmm14, zmm16
    vminps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_max:
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vmaxps zmm0, zmm0, zmm16
    vmaxps zmm1, zmm1, zmm16
    vmaxps zmm2, zmm2, zmm16
    vmaxps zmm3, zmm3, zmm16
    vmaxps zmm4, zmm4, zmm16
    vmaxps zmm5, zmm5, zmm16
    vmaxps zmm6, zmm6, zmm16
    vmaxps zmm7, zmm7, zmm16
    vmaxps zmm8, zmm8, zmm16
    vmaxps zmm9, zmm9, zmm16
    vmaxps zmm10, zmm10, zmm16
    vmaxps zmm11, zmm11, zmm16
    vmaxps zmm12, zmm12, zmm16
    vmaxps zmm13, zmm13, zmm16
    vmaxps zmm14, zmm14, zmm16
    vmaxps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_add:
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vaddps zmm0, zmm0, zmm16
    vaddps zmm1, zmm1, zmm16
    vaddps zmm2, zmm2, zmm16
    vaddps zmm3, zmm3, zmm16
    vaddps zmm4, zmm4, zmm16
    vaddps zmm5, zmm5, zmm16
    vaddps zmm6, zmm6, zmm16
    vaddps zmm7, zmm7, zmm16
    vaddps zmm8, zmm8, zmm16
    vaddps zmm9, zmm9, zmm16
    vaddps zmm10, zmm10, zmm16
    vaddps zmm11, zmm11, zmm16
    vaddps zmm12, zmm12, zmm16
    vaddps zmm13, zmm13, zmm16
    vaddps zmm14, zmm14, zmm16
    vaddps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_mul:
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vmulps zmm0, zmm0, zmm16
    vmulps zmm1, zmm1, zmm16
    vmulps zmm2, zmm2, zmm16
    vmulps zmm3, zmm3, zmm16
    vmulps zmm4, zmm4, zmm16
    vmulps zmm5, zmm5, zmm16
    vmulps zmm6, zmm6, zmm16
    vmulps zmm7, zmm7, zmm16
    vmulps zmm8, zmm8, zmm16
    vmulps zmm9, zmm9, zmm16
    vmulps zmm10, zmm10, zmm16
    vmulps zmm11, zmm11, zmm16
    vmulps zmm12, zmm12, zmm16
    vmulps zmm13, zmm13, zmm16
    vmulps zmm14, zmm14, zmm16
    vmulps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_sub:
    // non-flipped sub = operand - acc (matches fma_mmm_ymm_ops.j2 scalar macro)
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vsubps zmm0, zmm16, zmm0
    vsubps zmm1, zmm16, zmm1
    vsubps zmm2, zmm16, zmm2
    vsubps zmm3, zmm16, zmm3
    vsubps zmm4, zmm16, zmm4
    vsubps zmm5, zmm16, zmm5
    vsubps zmm6, zmm16, zmm6
    vsubps zmm7, zmm16, zmm7
    vsubps zmm8, zmm16, zmm8
    vsubps zmm9, zmm16, zmm9
    vsubps zmm10, zmm16, zmm10
    vsubps zmm11, zmm16, zmm11
    vsubps zmm12, zmm16, zmm12
    vsubps zmm13, zmm16, zmm13
    vsubps zmm14, zmm16, zmm14
    vsubps zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lscalar_sub_flipped:
    // flipped sub = acc - operand
    vbroadcastss    zmm16, dword ptr [rdi + 8]
    vsubps zmm0, zmm0, zmm16
    vsubps zmm1, zmm1, zmm16
    vsubps zmm2, zmm2, zmm16
    vsubps zmm3, zmm3, zmm16
    vsubps zmm4, zmm4, zmm16
    vsubps zmm5, zmm5, zmm16
    vsubps zmm6, zmm6, zmm16
    vsubps zmm7, zmm7, zmm16
    vsubps zmm8, zmm8, zmm16
    vsubps zmm9, zmm9, zmm16
    vsubps zmm10, zmm10, zmm16
    vsubps zmm11, zmm11, zmm16
    vsubps zmm12, zmm12, zmm16
    vsubps zmm13, zmm13, zmm16
    vsubps zmm14, zmm14, zmm16
    vsubps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lleaky_relu:
    // C[m, n] = (C[m, n] >= 0) ? C[m, n] : alpha * C[m, n]
    vbroadcastss    zmm17, dword ptr [rdi + 8]      // alpha
    vpxorq          zmm16, zmm16, zmm16             // 0.0
    
    vmulps          zmm18, zmm0, zmm17          // alpha * x
    vcmpps          k1, zmm0, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm0{k1}, zmm0, zmm18
    
    vmulps          zmm18, zmm1, zmm17          // alpha * x
    vcmpps          k1, zmm1, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm1{k1}, zmm1, zmm18
    
    vmulps          zmm18, zmm2, zmm17          // alpha * x
    vcmpps          k1, zmm2, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm2{k1}, zmm2, zmm18
    
    vmulps          zmm18, zmm3, zmm17          // alpha * x
    vcmpps          k1, zmm3, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm3{k1}, zmm3, zmm18
    
    vmulps          zmm18, zmm4, zmm17          // alpha * x
    vcmpps          k1, zmm4, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm4{k1}, zmm4, zmm18
    
    vmulps          zmm18, zmm5, zmm17          // alpha * x
    vcmpps          k1, zmm5, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm5{k1}, zmm5, zmm18
    
    vmulps          zmm18, zmm6, zmm17          // alpha * x
    vcmpps          k1, zmm6, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm6{k1}, zmm6, zmm18
    
    vmulps          zmm18, zmm7, zmm17          // alpha * x
    vcmpps          k1, zmm7, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm7{k1}, zmm7, zmm18
    
    vmulps          zmm18, zmm8, zmm17          // alpha * x
    vcmpps          k1, zmm8, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm8{k1}, zmm8, zmm18
    
    vmulps          zmm18, zmm9, zmm17          // alpha * x
    vcmpps          k1, zmm9, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm9{k1}, zmm9, zmm18
    
    vmulps          zmm18, zmm10, zmm17          // alpha * x
    vcmpps          k1, zmm10, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm10{k1}, zmm10, zmm18
    
    vmulps          zmm18, zmm11, zmm17          // alpha * x
    vcmpps          k1, zmm11, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm11{k1}, zmm11, zmm18
    
    vmulps          zmm18, zmm12, zmm17          // alpha * x
    vcmpps          k1, zmm12, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm12{k1}, zmm12, zmm18
    
    vmulps          zmm18, zmm13, zmm17          // alpha * x
    vcmpps          k1, zmm13, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm13{k1}, zmm13, zmm18
    
    vmulps          zmm18, zmm14, zmm17          // alpha * x
    vcmpps          k1, zmm14, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm14{k1}, zmm14, zmm18
    
    vmulps          zmm18, zmm15, zmm17          // alpha * x
    vcmpps          k1, zmm15, zmm16, 1          // imm 1 = LT (signed): 1 where x < 0
    vblendmps       zmm15{k1}, zmm15, zmm18
    
    jmp     .Lnon_linear_loop

.Lper_row_min:
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vminps          zmm0, zmm0, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vminps          zmm1, zmm1, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vminps          zmm2, zmm2, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vminps          zmm3, zmm3, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vminps          zmm4, zmm4, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vminps          zmm5, zmm5, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vminps          zmm6, zmm6, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vminps          zmm7, zmm7, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vminps          zmm8, zmm8, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vminps          zmm9, zmm9, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vminps          zmm10, zmm10, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vminps          zmm11, zmm11, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vminps          zmm12, zmm12, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vminps          zmm13, zmm13, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vminps          zmm14, zmm14, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vminps          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_max:
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vmaxps          zmm0, zmm0, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vmaxps          zmm1, zmm1, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vmaxps          zmm2, zmm2, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vmaxps          zmm3, zmm3, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vmaxps          zmm4, zmm4, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vmaxps          zmm5, zmm5, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vmaxps          zmm6, zmm6, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vmaxps          zmm7, zmm7, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vmaxps          zmm8, zmm8, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vmaxps          zmm9, zmm9, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vmaxps          zmm10, zmm10, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vmaxps          zmm11, zmm11, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vmaxps          zmm12, zmm12, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vmaxps          zmm13, zmm13, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vmaxps          zmm14, zmm14, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vmaxps          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_add:
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vaddps          zmm0, zmm0, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vaddps          zmm1, zmm1, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vaddps          zmm2, zmm2, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vaddps          zmm3, zmm3, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vaddps          zmm4, zmm4, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vaddps          zmm5, zmm5, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vaddps          zmm6, zmm6, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vaddps          zmm7, zmm7, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vaddps          zmm8, zmm8, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vaddps          zmm9, zmm9, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vaddps          zmm10, zmm10, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vaddps          zmm11, zmm11, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vaddps          zmm12, zmm12, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vaddps          zmm13, zmm13, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vaddps          zmm14, zmm14, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vaddps          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_mul:
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vmulps          zmm0, zmm0, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vmulps          zmm1, zmm1, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vmulps          zmm2, zmm2, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vmulps          zmm3, zmm3, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vmulps          zmm4, zmm4, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vmulps          zmm5, zmm5, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vmulps          zmm6, zmm6, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vmulps          zmm7, zmm7, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vmulps          zmm8, zmm8, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vmulps          zmm9, zmm9, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vmulps          zmm10, zmm10, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vmulps          zmm11, zmm11, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vmulps          zmm12, zmm12, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vmulps          zmm13, zmm13, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vmulps          zmm14, zmm14, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vmulps          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_sub:
    // non-flipped sub = operand - acc
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vsubps          zmm0, zmm16, zmm0
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vsubps          zmm1, zmm16, zmm1
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vsubps          zmm2, zmm16, zmm2
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vsubps          zmm3, zmm16, zmm3
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vsubps          zmm4, zmm16, zmm4
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vsubps          zmm5, zmm16, zmm5
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vsubps          zmm6, zmm16, zmm6
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vsubps          zmm7, zmm16, zmm7
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vsubps          zmm8, zmm16, zmm8
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vsubps          zmm9, zmm16, zmm9
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vsubps          zmm10, zmm16, zmm10
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vsubps          zmm11, zmm16, zmm11
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vsubps          zmm12, zmm16, zmm12
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vsubps          zmm13, zmm16, zmm13
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vsubps          zmm14, zmm16, zmm14
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vsubps          zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lper_row_sub_flipped:
    // flipped sub = acc - operand
    mov rax, [rdi + 8]
    vbroadcastss    zmm16, dword ptr [rax + 0]
    vsubps          zmm0, zmm0, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 4]
    vsubps          zmm1, zmm1, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 8]
    vsubps          zmm2, zmm2, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 12]
    vsubps          zmm3, zmm3, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 16]
    vsubps          zmm4, zmm4, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 20]
    vsubps          zmm5, zmm5, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 24]
    vsubps          zmm6, zmm6, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 28]
    vsubps          zmm7, zmm7, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 32]
    vsubps          zmm8, zmm8, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 36]
    vsubps          zmm9, zmm9, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 40]
    vsubps          zmm10, zmm10, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 44]
    vsubps          zmm11, zmm11, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 48]
    vsubps          zmm12, zmm12, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 52]
    vsubps          zmm13, zmm13, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 56]
    vsubps          zmm14, zmm14, zmm16
    vbroadcastss    zmm16, dword ptr [rax + 60]
    vsubps          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_min:
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vminps zmm0, zmm0, zmm16
    vminps zmm1, zmm1, zmm16
    vminps zmm2, zmm2, zmm16
    vminps zmm3, zmm3, zmm16
    vminps zmm4, zmm4, zmm16
    vminps zmm5, zmm5, zmm16
    vminps zmm6, zmm6, zmm16
    vminps zmm7, zmm7, zmm16
    vminps zmm8, zmm8, zmm16
    vminps zmm9, zmm9, zmm16
    vminps zmm10, zmm10, zmm16
    vminps zmm11, zmm11, zmm16
    vminps zmm12, zmm12, zmm16
    vminps zmm13, zmm13, zmm16
    vminps zmm14, zmm14, zmm16
    vminps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_max:
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vmaxps zmm0, zmm0, zmm16
    vmaxps zmm1, zmm1, zmm16
    vmaxps zmm2, zmm2, zmm16
    vmaxps zmm3, zmm3, zmm16
    vmaxps zmm4, zmm4, zmm16
    vmaxps zmm5, zmm5, zmm16
    vmaxps zmm6, zmm6, zmm16
    vmaxps zmm7, zmm7, zmm16
    vmaxps zmm8, zmm8, zmm16
    vmaxps zmm9, zmm9, zmm16
    vmaxps zmm10, zmm10, zmm16
    vmaxps zmm11, zmm11, zmm16
    vmaxps zmm12, zmm12, zmm16
    vmaxps zmm13, zmm13, zmm16
    vmaxps zmm14, zmm14, zmm16
    vmaxps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_add:
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vaddps zmm0, zmm0, zmm16
    vaddps zmm1, zmm1, zmm16
    vaddps zmm2, zmm2, zmm16
    vaddps zmm3, zmm3, zmm16
    vaddps zmm4, zmm4, zmm16
    vaddps zmm5, zmm5, zmm16
    vaddps zmm6, zmm6, zmm16
    vaddps zmm7, zmm7, zmm16
    vaddps zmm8, zmm8, zmm16
    vaddps zmm9, zmm9, zmm16
    vaddps zmm10, zmm10, zmm16
    vaddps zmm11, zmm11, zmm16
    vaddps zmm12, zmm12, zmm16
    vaddps zmm13, zmm13, zmm16
    vaddps zmm14, zmm14, zmm16
    vaddps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_mul:
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vmulps zmm0, zmm0, zmm16
    vmulps zmm1, zmm1, zmm16
    vmulps zmm2, zmm2, zmm16
    vmulps zmm3, zmm3, zmm16
    vmulps zmm4, zmm4, zmm16
    vmulps zmm5, zmm5, zmm16
    vmulps zmm6, zmm6, zmm16
    vmulps zmm7, zmm7, zmm16
    vmulps zmm8, zmm8, zmm16
    vmulps zmm9, zmm9, zmm16
    vmulps zmm10, zmm10, zmm16
    vmulps zmm11, zmm11, zmm16
    vmulps zmm12, zmm12, zmm16
    vmulps zmm13, zmm13, zmm16
    vmulps zmm14, zmm14, zmm16
    vmulps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_sub:
    // non-flipped sub = operand - acc
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vsubps zmm0, zmm16, zmm0
    vsubps zmm1, zmm16, zmm1
    vsubps zmm2, zmm16, zmm2
    vsubps zmm3, zmm16, zmm3
    vsubps zmm4, zmm16, zmm4
    vsubps zmm5, zmm16, zmm5
    vsubps zmm6, zmm16, zmm6
    vsubps zmm7, zmm16, zmm7
    vsubps zmm8, zmm16, zmm8
    vsubps zmm9, zmm16, zmm9
    vsubps zmm10, zmm16, zmm10
    vsubps zmm11, zmm16, zmm11
    vsubps zmm12, zmm16, zmm12
    vsubps zmm13, zmm16, zmm13
    vsubps zmm14, zmm16, zmm14
    vsubps zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lper_col_sub_flipped:
    // flipped sub = acc - operand
    mov     rax, [rdi + 8]
    vmovups         zmm16, [rax]
    vsubps zmm0, zmm0, zmm16
    vsubps zmm1, zmm1, zmm16
    vsubps zmm2, zmm2, zmm16
    vsubps zmm3, zmm3, zmm16
    vsubps zmm4, zmm4, zmm16
    vsubps zmm5, zmm5, zmm16
    vsubps zmm6, zmm6, zmm16
    vsubps zmm7, zmm7, zmm16
    vsubps zmm8, zmm8, zmm16
    vsubps zmm9, zmm9, zmm16
    vsubps zmm10, zmm10, zmm16
    vsubps zmm11, zmm11, zmm16
    vsubps zmm12, zmm12, zmm16
    vsubps zmm13, zmm13, zmm16
    vsubps zmm14, zmm14, zmm16
    vsubps zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lload_tile:
    // Scratch layout is COL-MAJOR f32 (col_byte_stride = item_size * MR =
    // 4 * 16 = 64): tile[col][row] at offset col*64 + row*4. Gather row m's
    // 16 cols at index step 64.
    mov         r8, [rdi + 8]
    vmovdqa32   zmm16, [rip + .Llane_offsets_64]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm0{k1}, [r8 + zmm16 + 0]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm1{k1}, [r8 + zmm16 + 4]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm2{k1}, [r8 + zmm16 + 8]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm3{k1}, [r8 + zmm16 + 12]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm4{k1}, [r8 + zmm16 + 16]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm5{k1}, [r8 + zmm16 + 20]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm6{k1}, [r8 + zmm16 + 24]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm7{k1}, [r8 + zmm16 + 28]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm8{k1}, [r8 + zmm16 + 32]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm9{k1}, [r8 + zmm16 + 36]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm10{k1}, [r8 + zmm16 + 40]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm11{k1}, [r8 + zmm16 + 44]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm12{k1}, [r8 + zmm16 + 48]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm13{k1}, [r8 + zmm16 + 52]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm14{k1}, [r8 + zmm16 + 56]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm15{k1}, [r8 + zmm16 + 60]
    
    jmp     .Lnon_linear_loop

.Ladd_unicast:
    mov     r10,    [rdi + 8]           // c ptr (base)
    mov     rsi,    [rdi + 16]          // row stride
    mov     rbx,    [rdi + 24]          // col stride
    mov     r8,     [rdi + 32]          // item size (4 for f32)

    cmp     r8, 4
    jne     .Lunsupported            // f32 kernel: only item_size=4

    // i32-strided gather (f32 same bit-width: vpgatherdd is correct).
    mov             eax, ebx
    vmovd           xmm16, eax
    vpbroadcastd    zmm16, xmm16
    vpmulld         zmm16, zmm16, [rip + .Llane_indices]

    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm0, zmm0, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm1, zmm1, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm2, zmm2, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm3, zmm3, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm4, zmm4, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm5, zmm5, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm6, zmm6, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm7, zmm7, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm8, zmm8, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm9, zmm9, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm10, zmm10, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm11, zmm11, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm12, zmm12, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm13, zmm13, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm14, zmm14, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vaddps          zmm15, zmm15, zmm17
    add             r10, rsi
    
    jmp     .Lnon_linear_loop

.Ladd_row_col_products:
    // bias[m, n] = row_data[m] * col_data[n], FMA-add to C[m, n].
    mov             rax, [rdi + 8]
    mov             rbx, [rdi + 16]

    vmovups         zmm16, [rbx]                          // 16 col_data values

    
    vbroadcastss    zmm17, dword ptr [rax + 0]    // splat row_data[m]
    vfmadd231ps     zmm0, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 4]    // splat row_data[m]
    vfmadd231ps     zmm1, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 8]    // splat row_data[m]
    vfmadd231ps     zmm2, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 12]    // splat row_data[m]
    vfmadd231ps     zmm3, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 16]    // splat row_data[m]
    vfmadd231ps     zmm4, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 20]    // splat row_data[m]
    vfmadd231ps     zmm5, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 24]    // splat row_data[m]
    vfmadd231ps     zmm6, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 28]    // splat row_data[m]
    vfmadd231ps     zmm7, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 32]    // splat row_data[m]
    vfmadd231ps     zmm8, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 36]    // splat row_data[m]
    vfmadd231ps     zmm9, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 40]    // splat row_data[m]
    vfmadd231ps     zmm10, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 44]    // splat row_data[m]
    vfmadd231ps     zmm11, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 48]    // splat row_data[m]
    vfmadd231ps     zmm12, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 52]    // splat row_data[m]
    vfmadd231ps     zmm13, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 56]    // splat row_data[m]
    vfmadd231ps     zmm14, zmm17, zmm16
    
    vbroadcastss    zmm17, dword ptr [rax + 60]    // splat row_data[m]
    vfmadd231ps     zmm15, zmm17, zmm16
    
    jmp     .Lnon_linear_loop

// ---- q_scale / q_shr / q_shl: not meaningful for f32, stub to unsupported.
.Lq_scale:
.Lq_shl:
.Lq_shr:
    jmp     .Lunsupported

// ---- Store ---------------------------------------------------------------

.Lstore:
    mov     r8,     [rdi + 8]               // c ptr
    mov     rsi,    [rdi + 16]              // row stride
    mov     rdx,    [rdi + 24]              // col stride
    mov     rcx,    [rdi + 32]              // item size

    cmp     rcx, 4
    jne     .Lunsupported                // f32 kernel: only item_size=4

    cmp     rdx, 4
    je      .Lstore_strides_f32_row_contig

    // Generic f32 strided store
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm0, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm1, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm2, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm3, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm4, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm5, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm6, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm7, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm8, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm9, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm10, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm11, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm12, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm13, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm14, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm15, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    jmp     .Lnon_linear_loop

.Lstore_strides_f32_row_contig:
    // C is row-major in memory: each row's 16 f32 are contiguous; one
    // 64-byte vmovups per row.
    
    vmovups     [r8], zmm0
    add         r8, rsi
    
    vmovups     [r8], zmm1
    add         r8, rsi
    
    vmovups     [r8], zmm2
    add         r8, rsi
    
    vmovups     [r8], zmm3
    add         r8, rsi
    
    vmovups     [r8], zmm4
    add         r8, rsi
    
    vmovups     [r8], zmm5
    add         r8, rsi
    
    vmovups     [r8], zmm6
    add         r8, rsi
    
    vmovups     [r8], zmm7
    add         r8, rsi
    
    vmovups     [r8], zmm8
    add         r8, rsi
    
    vmovups     [r8], zmm9
    add         r8, rsi
    
    vmovups     [r8], zmm10
    add         r8, rsi
    
    vmovups     [r8], zmm11
    add         r8, rsi
    
    vmovups     [r8], zmm12
    add         r8, rsi
    
    vmovups     [r8], zmm13
    add         r8, rsi
    
    vmovups     [r8], zmm14
    add         r8, rsi
    
    vmovups     [r8], zmm15
    add         r8, rsi
    
    jmp     .Lnon_linear_loop

.Lreturn:
    tilerelease
    add         rsp, 64

    ldmxcsr     [rsp + 4]
    add         rsp, 8

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx



    mov rsp, rbp
    pop rbp
    ret

// ---- Read-only data (RIP-relative) ---------------------------------------

.p2align 6
.Llane_offsets_64:
    .int  0,  64, 128, 192, 256, 320, 384, 448
    .int  512, 576, 640, 704, 768, 832, 896, 960

.p2align 6
.Llane_indices:
    .int  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15


.cfi_endproc
