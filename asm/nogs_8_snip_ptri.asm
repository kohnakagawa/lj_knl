# -- Begin  _Z21force_sorted_z_intrinv
	.text
# mark_begin;
# Threads 2
        .align    16,0x90
	.globl _Z21force_sorted_z_intrinv
# --- force_sorted_z_intrin()
_Z21force_sorted_z_intrinv:
..B12.1:                        # Preds ..B12.0
                                # Execution count [1.00e+00]
..L495:
                # optimization report
                # ループが融合されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 6.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4002:
	.loc    1  2043  is_stmt 1
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z21force_sorted_z_intrinv.489:
..L490:
                                                        #2041.29
..LN4003:
	.loc    1  2041  is_stmt 1
        pushq     %rbp                                          #2041.29
	.cfi_def_cfa_offset 16
..LN4004:
        movq      %rsp, %rbp                                    #2041.29
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
..LN4005:
        andq      $-64, %rsp                                    #2041.29
..LN4006:
        subq      $2112, %rsp                                   #2041.29 c1
..LN4007:
	.loc    1  2042  prologue_end  is_stmt 1
        movslq    particle_number(%rip), %rdi                   #2042.18 c1
..LN4008:
	.loc    1  2043  is_stmt 1
        vpxord    %zmm23, %zmm23, %zmm23                        #2043.22 c1
..LN4009:
	.loc    1  2044  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.46(%rip), %zmm11        #2044.21 c1
..LN4010:
	.loc    1  2058  is_stmt 1
        xorl      %r8d, %r8d                                    #2058.3 c1
..LN4011:
        xorl      %esi, %esi                                    #2058.3 c3
..LN4012:
	.loc    1  2045  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.50(%rip), %zmm10        #2045.21 c5
..LN4013:
	.loc    1  2046  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.51(%rip), %zmm9         #2046.21 c7
..LN4014:
	.loc    1  2048  is_stmt 1
        vmovups   .L_2il0floatpacket.57(%rip), %zmm8            #2048.25 c11 stall 1
..LN4015:
	.loc    1  2049  is_stmt 1
        vmovups   .L_2il0floatpacket.58(%rip), %zmm7            #2049.25 c13
..LN4016:
	.loc    1  2050  is_stmt 1
        vmovups   .L_2il0floatpacket.59(%rip), %zmm6            #2050.27 c17 stall 1
..LN4017:
	.loc    1  2051  is_stmt 1
        vmovups   .L_2il0floatpacket.60(%rip), %zmm5            #2051.27 c19
..LN4018:
	.loc    1  2052  is_stmt 1
        vmovups   .L_2il0floatpacket.61(%rip), %zmm4            #2052.27 c23 stall 1
..LN4019:
	.loc    1  2053  is_stmt 1
        vmovups   .L_2il0floatpacket.62(%rip), %zmm3            #2053.27 c25
..LN4020:
	.loc    1  2054  is_stmt 1
        vmovups   .L_2il0floatpacket.63(%rip), %zmm15           #2054.28 c29 stall 1
..LN4021:
	.loc    1  2055  is_stmt 1
        vmovups   .L_2il0floatpacket.64(%rip), %zmm2            #2055.24 c31
..LN4022:
	.loc    1  2056  is_stmt 1
        vmovups   .L_2il0floatpacket.65(%rip), %zmm1            #2056.24 c35 stall 1
..LN4023:
	.loc    1  2058  is_stmt 1
        testq     %rdi, %rdi                                    #2058.23 c35
..LN4024:
        jle       ..B12.31      # Prob 9%                       #2058.23 c37
..LN4025:
                                # LOE rbx rsi rdi r8 r12 r13 r14 r15 zmm1 zmm2 zmm3 zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm15 zmm23
..B12.2:                        # Preds ..B12.1
                                # Execution count [9.00e-01]
..LN4026:
	.loc    1  2146  is_stmt 1
        movq      %r12, 1664(%rsp)                              #2146.16[spill] c1
..LN4027:
        movq      %r13, 1672(%rsp)                              #2146.16[spill] c1
..LN4028:
        movq      %r14, 1680(%rsp)                              #2146.16[spill] c3
..LN4029:
        movq      %r15, 1688(%rsp)                              #2146.16[spill] c3
..LN4030:
        movq      %rbx, 1696(%rsp)                              #2146.16[spill] c5
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xfe, 0xff, 0xff, 0x22
..LN4031:
                                # LOE rsi rdi r8 zmm15 zmm23
..B12.3:                        # Preds ..B12.29 ..B12.2
                                # Execution count [7.20e+00]
..L503:
                # optimization report
                # ループが融合されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4032:
	.loc    1  2062  is_stmt 1
..L502:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4033:
	.loc    1  2061  is_stmt 1
..L501:
                # optimization report
                # ユーザー定義のベクトル組込み関数を含むループ
                # %s はベクトル化されませんでした: 内部ループがすでにベクトル化されています。
..LN4034:
	.loc    1  2058  is_stmt 1
..LN4035:
	.loc    1  2059  is_stmt 1
        movq      number_of_partners(%rip), %r12                #2059.20 c1
..LN4036:
	.loc    1  2060  is_stmt 1
        movq      pointer(%rip), %rcx                           #2060.20 c1
..LN4037:
	.loc    1  2063  is_stmt 1
        vmovaps   %zmm23, %zmm13                                #2063.16 c1
..LN4038:
	.loc    1  2064  is_stmt 1
        xorl      %edx, %edx                                    #2064.5 c1
..LN4039:
	.loc    1  2059  is_stmt 1
        movl      (%r12,%r8,4), %r12d                           #2059.20 c5 stall 1
..LN4040:
	.loc    1  2060  is_stmt 1
        movslq    (%rcx,%r8,4), %rax                            #2060.20 c5
..LN4041:
	.loc    1  2064  is_stmt 1
        movl      %r12d, %r10d                                  #2064.29 c9 stall 1
..LN4042:
	.loc    1  2060  is_stmt 1
        movq      sorted_list(%rip), %rcx                       #2060.55 c9
..LN4043:
	.loc    1  2061  is_stmt 1
        vmovups   z(%rsi), %zmm12                               #2061.42 c9
..LN4044:
	.loc    1  2064  is_stmt 1
        sarl      $2, %r10d                                     #2064.29 c11
..LN4045:
        shrl      $29, %r10d                                    #2064.29 c13
..LN4046:
	.loc    1  2060  is_stmt 1
        lea       (%rcx,%rax,4), %rbx                           #2060.55 c13
..LN4047:
	.loc    1  2064  is_stmt 1
        addl      %r12d, %r10d                                  #2064.29 c15
..LN4048:
	.loc    1  2060  is_stmt 1
        movq      %rbx, %r11                                    #2060.55 c15
..LN4049:
	.loc    1  2062  is_stmt 1
        vpermpd   %zmm12, %zmm15, %zmm11                        #2062.16 c15
..LN4050:
	.loc    1  2064  is_stmt 1
        sarl      $3, %r10d                                     #2064.29 c17
..LN4051:
        lea       (,%r10,8), %r9d                               #2064.31 c19
..LN4052:
        testl     %r9d, %r9d                                    #2064.31 c21
..LN4053:
        jle       ..B12.9       # Prob 10%                      #2064.31 c23
..LN4054:
                                # LOE rcx rbx rsi rdi r8 r11 edx r9d r10d r12d zmm11 zmm12 zmm13 zmm15 zmm23
..B12.4:                        # Preds ..B12.3
                                # Execution count [6.48e+00]
..LN4055:
	.loc    1  2128  is_stmt 1
        movl      $240, %eax                                    #2128.7 c1
..LN4056:
	.loc    1  2064  is_stmt 1
        movq      %r11, 2048(%rsp)                              #2064.31[spill] c1
..LN4057:
        vpxorq    %zmm10, %zmm10, %zmm10                        #2064.31 c1
..LN4058:
        movl      %r12d, 2056(%rsp)                             #2064.31[spill] c1
..LN4059:
	.loc    1  2128  is_stmt 1
        kmovw     %eax, %k1                                     #2128.7 c3
..LN4060:
	.loc    1  2064  is_stmt 1
        movq      %rsi, 2064(%rsp)                              #2064.31[spill] c3
..LN4061:
        lea       7(,%r10,8), %eax                              #2064.31 c3
..LN4062:
        movq      %r8, 2072(%rsp)                               #2064.31[spill] c5
..LN4063:
        sarl      $2, %eax                                      #2064.31 c5
..LN4064:
        movq      %rdi, 2080(%rsp)                              #2064.31[spill] c5
..LN4065:
        shrl      $29, %eax                                     #2064.31 c7
..LN4066:
        vbroadcastsd .L_2il0floatpacket.23(%rip), %zmm14        #2064.31 c7
..LN4067:
        vmovups   .L_2il0floatpacket.65(%rip), %zmm16           #2064.31 c7
..LN4068:
        lea       7(%rax,%r10,8), %eax                          #2064.31 c13 stall 2
..LN4069:
        vmovups   .L_2il0floatpacket.64(%rip), %zmm17           #2064.31 c13
..LN4070:
        sarl      $3, %eax                                      #2064.31 c15
..LN4071:
        vmovups   .L_2il0floatpacket.62(%rip), %zmm18           #2064.31 c15
..LN4072:
        vmovups   .L_2il0floatpacket.61(%rip), %zmm19           #2064.31 c19 stall 1
..LN4073:
        vmovups   .L_2il0floatpacket.60(%rip), %zmm20           #2064.31 c21
..LN4074:
        vmovups   .L_2il0floatpacket.59(%rip), %zmm21           #2064.31 c25 stall 1
..LN4075:
        vmovups   .L_2il0floatpacket.58(%rip), %zmm22           #2064.31 c27
..LN4076:
        vmovups   .L_2il0floatpacket.57(%rip), %zmm24           #2064.31 c31 stall 1
..LN4077:
        vbroadcastsd .L_2il0floatpacket.51(%rip), %zmm25        #2064.31 c33
..LN4078:
        vbroadcastsd .L_2il0floatpacket.50(%rip), %zmm26        #2064.31 c37 stall 1
..LN4079:
        vbroadcastsd .L_2il0floatpacket.46(%rip), %zmm27        #2064.31 c39
..LN4080:
                                # LOE rcx rbx eax edx r9d r10d zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 k1
..B12.5:                        # Preds ..B12.7 ..B12.4
                                # Execution count [3.60e+01]
..L520:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 8.562500
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4081:
	.loc    1  2104  is_stmt 1
..L519:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4082:
	.loc    1  2103  is_stmt 1
..L518:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4083:
	.loc    1  2102  is_stmt 1
..L517:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4084:
	.loc    1  2100  is_stmt 1
..L516:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4085:
	.loc    1  2099  is_stmt 1
..L515:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4086:
	.loc    1  2098  is_stmt 1
..L514:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4087:
	.loc    1  2097  is_stmt 1
..L513:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.718750
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4088:
	.loc    1  2094  is_stmt 1
..L512:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4089:
	.loc    1  2093  is_stmt 1
..L511:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4090:
	.loc    1  2088  is_stmt 1
..L510:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4091:
	.loc    1  2087  is_stmt 1
..L509:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4092:
	.loc    1  2082  is_stmt 1
..L508:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4093:
	.loc    1  2081  is_stmt 1
..L507:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4094:
	.loc    1  2076  is_stmt 1
..L506:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.867188
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4095:
	.loc    1  2075  is_stmt 1
..L505:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 6.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4096:
	.loc    1  2073  is_stmt 1
..L504:
                # optimization report
                # ユーザー定義のベクトル組込み関数を含むループ
                # %s はベクトル化されませんでした: 内部ループがすでにベクトル化されています。
..LN4097:
	.loc    1  2064  is_stmt 1
..LN4098:
	.loc    1  2110  is_stmt 1
        vmovupd   %zmm27, 1984(%rsp)                            #2110.46 c1
..LN4099:
	.loc    1  2065  is_stmt 1
        movslq    (%rbx), %r14                                  #2065.36 c1
..LN4100:
	.loc    1  2075  is_stmt 1
        vmovaps   %zmm24, %zmm8                                 #2075.20 c1
..LN4101:
	.loc    1  2081  is_stmt 1
        vmovaps   %zmm24, %zmm5                                 #2081.20 c1
..LN4102:
	.loc    1  2087  is_stmt 1
        vmovaps   %zmm24, %zmm3                                 #2087.20 c3
..LN4103:
	.loc    1  2093  is_stmt 1
        vmovaps   %zmm24, %zmm28                                #2093.20 c3
..LN4104:
	.loc    1  2066  is_stmt 1
        movslq    4(%rbx), %r13                                 #2066.36 c5
..LN4105:
	.loc    1  2065  is_stmt 1
        movslq    (%rcx,%r14,4), %r14                           #2065.23 c7
..LN4106:
	.loc    1  2066  is_stmt 1
        movslq    (%rcx,%r13,4), %r13                           #2066.23 c9
..LN4107:
	.loc    1  2067  is_stmt 1
        movslq    8(%rbx), %r12                                 #2067.36 c11
..LN4108:
	.loc    1  2073  is_stmt 1
        shlq      $6, %r14                                      #2073.46 c11
..LN4109:
	.loc    1  2068  is_stmt 1
        movslq    12(%rbx), %r11                                #2068.36 c13
..LN4110:
	.loc    1  2074  is_stmt 1
        shlq      $6, %r13                                      #2074.46 c13
..LN4111:
	.loc    1  2067  is_stmt 1
        movslq    (%rcx,%r12,4), %r12                           #2067.23 c15
..LN4112:
	.loc    1  2068  is_stmt 1
        movslq    (%rcx,%r11,4), %r11                           #2068.23 c17
..LN4113:
	.loc    1  2073  is_stmt 1
        vmovups   z(%r14), %zmm9                                #2073.46 c19
..LN4114:
	.loc    1  2079  is_stmt 1
        shlq      $6, %r12                                      #2079.46 c19
..LN4115:
	.loc    1  2074  is_stmt 1
        vmovups   z(%r13), %zmm7                                #2074.46 c21
..LN4116:
	.loc    1  2080  is_stmt 1
        shlq      $6, %r11                                      #2080.46 c21
..LN4117:
	.loc    1  2069  is_stmt 1
        movslq    16(%rbx), %r8                                 #2069.36 c25 stall 1
..LN4118:
	.loc    1  2070  is_stmt 1
        movslq    20(%rbx), %rdi                                #2070.36 c27
..LN4119:
	.loc    1  2075  is_stmt 1
        vpermi2pd %zmm7, %zmm9, %zmm8                           #2075.20 c27
..LN4120:
	.loc    1  2076  is_stmt 1
        vpermt2pd %zmm7, %zmm22, %zmm9                          #2076.20 c27
..LN4121:
	.loc    1  2079  is_stmt 1
        vmovups   z(%r12), %zmm7                                #2079.46 c29
..LN4122:
	.loc    1  2077  is_stmt 1
        vsubpd    %zmm11, %zmm8, %zmm8                          #2077.30 c29
..LN4123:
	.loc    1  2080  is_stmt 1
        vmovups   z(%r11), %zmm4                                #2080.46 c31
..LN4124:
	.loc    1  2069  is_stmt 1
        movslq    (%rcx,%r8,4), %r8                             #2069.23 c35 stall 1
..LN4125:
	.loc    1  2070  is_stmt 1
        movslq    (%rcx,%rdi,4), %rdi                           #2070.23 c37
..LN4126:
	.loc    1  2081  is_stmt 1
        vpermi2pd %zmm4, %zmm7, %zmm5                           #2081.20 c37
..LN4127:
	.loc    1  2082  is_stmt 1
        vpermt2pd %zmm4, %zmm22, %zmm7                          #2082.20 c37
..LN4128:
	.loc    1  2071  is_stmt 1
        movslq    24(%rbx), %rsi                                #2071.36 c39
..LN4129:
	.loc    1  2085  is_stmt 1
        shlq      $6, %r8                                       #2085.46 c39
..LN4130:
	.loc    1  2083  is_stmt 1
        vsubpd    %zmm11, %zmm5, %zmm6                          #2083.30 c39
..LN4131:
	.loc    1  2072  is_stmt 1
        movslq    28(%rbx), %r15                                #2072.36 c41
..LN4132:
	.loc    1  2086  is_stmt 1
        shlq      $6, %rdi                                      #2086.46 c41
..LN4133:
	.loc    1  2071  is_stmt 1
        movslq    (%rcx,%rsi,4), %rsi                           #2071.23 c43
..LN4134:
	.loc    1  2072  is_stmt 1
        movslq    (%rcx,%r15,4), %r15                           #2072.23 c45
..LN4135:
	.loc    1  2098  is_stmt 1
        vunpckhpd %zmm6, %zmm8, %zmm0                           #2098.19 c45
..LN4136:
	.loc    1  2097  is_stmt 1
        vunpcklpd %zmm6, %zmm8, %zmm4                           #2097.19 c45
..LN4137:
	.loc    1  2085  is_stmt 1
        vmovups   z(%r8), %zmm5                                 #2085.46 c47
..LN4138:
	.loc    1  2091  is_stmt 1
        shlq      $6, %rsi                                      #2091.46 c47
..LN4139:
	.loc    1  2086  is_stmt 1
        vmovups   z(%rdi), %zmm2                                #2086.46 c49
..LN4140:
	.loc    1  2092  is_stmt 1
        shlq      $6, %r15                                      #2092.46 c49
..LN4141:
        vmovups   z(%r15), %zmm1                                #2092.46 c53 stall 1
..LN4142:
	.loc    1  2087  is_stmt 1
        vpermi2pd %zmm2, %zmm5, %zmm3                           #2087.20 c55
..LN4143:
	.loc    1  2088  is_stmt 1
        vpermt2pd %zmm2, %zmm22, %zmm5                          #2088.20 c55
..LN4144:
	.loc    1  2091  is_stmt 1
        vmovups   z(%rsi), %zmm2                                #2091.46 c55
..LN4145:
	.loc    1  2089  is_stmt 1
        vsubpd    %zmm11, %zmm3, %zmm3                          #2089.30 c57
..LN4146:
	.loc    1  2093  is_stmt 1
        vpermi2pd %zmm1, %zmm2, %zmm28                          #2093.20 c61 stall 1
..LN4147:
	.loc    1  2094  is_stmt 1
        vpermt2pd %zmm1, %zmm22, %zmm2                          #2094.20 c63
..LN4148:
	.loc    1  2095  is_stmt 1
        vsubpd    %zmm11, %zmm28, %zmm1                         #2095.30 c63
..LN4149:
	.loc    1  2100  is_stmt 1
        vunpckhpd %zmm1, %zmm3, %zmm29                          #2100.19 c69 stall 2
..LN4150:
	.loc    1  2099  is_stmt 1
        vunpcklpd %zmm1, %zmm3, %zmm30                          #2099.19 c69
..LN4151:
	.loc    1  2103  is_stmt 1
        vpermt2pd %zmm29, %zmm17, %zmm0                         #2103.18 c71
..LN4152:
	.loc    1  2106  is_stmt 1
        vmulpd    %zmm0, %zmm0, %zmm31                          #2106.32 c73
..LN4153:
	.loc    1  2102  is_stmt 1
        vmovaps   %zmm17, %zmm0                                 #2102.18 c73
..LN4154:
        vpermi2pd %zmm30, %zmm4, %zmm0                          #2102.18 c75
..LN4155:
	.loc    1  2104  is_stmt 1
        vpermt2pd %zmm30, %zmm16, %zmm4                         #2104.18 c77
..LN4156:
	.loc    1  2106  is_stmt 1
        vfmadd213pd %zmm31, %zmm0, %zmm0                        #2106.32 c79
..LN4157:
        vfmadd213pd %zmm0, %zmm4, %zmm4                         #2106.42 c85 stall 2
..LN4158:
	.loc    1  2108  is_stmt 1
        vmovaps   %zmm25, %zmm0                                 #2108.32 c85
..LN4159:
	.loc    1  2107  is_stmt 1
        vmulpd    %zmm4, %zmm4, %zmm28                          #2107.24 c91 stall 2
..LN4160:
        vmulpd    %zmm28, %zmm4, %zmm29                         #2107.30 c97 stall 2
..LN4161:
	.loc    1  2108  is_stmt 1
        vmovaps   %zmm14, %zmm28                                #2108.53 c97
..LN4162:
        vmulpd    %zmm29, %zmm29, %zmm30                        #2108.47 c103 stall 2
..LN4163:
        vfmsub231pd %zmm29, %zmm26, %zmm0                       #2108.32 c103
..LN4164:
        vmulpd    %zmm30, %zmm4, %zmm30                         #2108.53 c109 stall 2
..LN4165:
        vrcp28pd  {sae}, %zmm30, %zmm29                         #2108.53 c115 stall 2
..LN4166:
        vfnmadd231pd {rn-sae}, %zmm30, %zmm29, %zmm28           #2108.53 c123 stall 3
..LN4167:
        vfmadd213pd {rn-sae}, %zmm29, %zmm28, %zmm29            #2108.53 c129 stall 2
..LN4168:
        vcmppd    $8, %zmm10, %zmm29, %k2                       #2108.53 c135 stall 2
..LN4169:
        vmulpd    %zmm29, %zmm0, %zmm28                         #2108.53 c135
..LN4170:
        kortestw  %k2, %k2                                      #2108.53 c137
..LN4171:
        je        ..B12.7       # Prob 25%                      #2108.53 c139
..LN4172:
                                # LOE rcx rbx rsi rdi r8 r11 r12 r13 r14 r15 eax edx r9d r10d zmm0 zmm1 zmm2 zmm3 zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm30 k1 k2
..B12.6:                        # Preds ..B12.5
                                # Execution count [2.70e+01]
..LN4173:
        vdivpd    %zmm30, %zmm0, %zmm28{%k2}                    #2108.53 c3 stall 1
..LN4174:
                                # LOE rcx rbx rsi rdi r8 r11 r12 r13 r14 r15 eax edx r9d r10d zmm1 zmm2 zmm3 zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 k1
..B12.7:                        # Preds ..B12.6 ..B12.5
                                # Execution count [3.60e+01]
..L537:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4175:
	.loc    1  2135  is_stmt 1
..L536:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4176:
	.loc    1  2134  is_stmt 1
..L535:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4177:
	.loc    1  2133  is_stmt 1
..L534:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4178:
	.loc    1  2132  is_stmt 1
..L533:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4179:
	.loc    1  2131  is_stmt 1
..L532:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4180:
	.loc    1  2130  is_stmt 1
..L531:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4181:
	.loc    1  2129  is_stmt 1
..L530:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4182:
	.loc    1  2127  is_stmt 1
..L529:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4183:
	.loc    1  2126  is_stmt 1
..L528:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4184:
	.loc    1  2125  is_stmt 1
..L527:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4185:
	.loc    1  2124  is_stmt 1
..L526:
                # optimization report
                # ループ文の順序が変更されました
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.195312
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4186:
	.loc    1  2115  is_stmt 1
..L525:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4187:
	.loc    1  2114  is_stmt 1
..L524:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4188:
	.loc    1  2113  is_stmt 1
..L523:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4189:
	.loc    1  2112  is_stmt 1
..L522:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4190:
	.loc    1  2111  is_stmt 1
..L521:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 9.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4191:
..LN4192:
	.loc    1  2110  is_stmt 1
        vcmppd    $14, 1984(%rsp), %zmm4, %k2                   #2110.23 c1
..LN4193:
	.loc    1  2064  is_stmt 1
        addl      $1, %edx                                      #2064.5 c1
..LN4194:
	.loc    1  2065  is_stmt 1
        addq      $32, %rbx                                     #2065.36 c1
..LN4195:
	.loc    1  2111  is_stmt 1
        vblendmpd %zmm23, %zmm28, %zmm0{%k2}                    #2111.13 c75 stall 36
..LN4196:
	.loc    1  2115  is_stmt 1
        vpermpd   %zmm0, %zmm18, %zmm4                          #2115.21 c77
..LN4197:
	.loc    1  2112  is_stmt 1
        vpermpd   %zmm0, %zmm21, %zmm28                         #2112.21 c77
..LN4198:
	.loc    1  2119  is_stmt 1
        vfnmadd231pd %zmm4, %zmm1, %zmm2                        #2119.7 c79
..LN4199:
	.loc    1  2123  is_stmt 1
        vmulpd    %zmm1, %zmm4, %zmm1                           #2123.7 c79
..LN4200:
	.loc    1  2116  is_stmt 1
        vfnmadd231pd %zmm28, %zmm8, %zmm9                       #2116.7 c85 stall 2
..LN4201:
	.loc    1  2123  is_stmt 1
        vfmadd213pd %zmm1, %zmm8, %zmm28                        #2123.7 c85
..LN4202:
	.loc    1  2114  is_stmt 1
        vpermpd   %zmm0, %zmm19, %zmm29                         #2114.21 c91 stall 2
..LN4203:
	.loc    1  2113  is_stmt 1
        vpermpd   %zmm0, %zmm20, %zmm30                         #2113.21 c91
..LN4204:
	.loc    1  2118  is_stmt 1
        vfnmadd231pd %zmm29, %zmm3, %zmm5                       #2118.7 c93
..LN4205:
	.loc    1  2123  is_stmt 1
        vfmadd213pd %zmm28, %zmm3, %zmm29                       #2123.7 c93
..LN4206:
	.loc    1  2117  is_stmt 1
        vfnmadd231pd %zmm30, %zmm6, %zmm7                       #2117.7 c99 stall 2
..LN4207:
	.loc    1  2123  is_stmt 1
        vfmadd213pd %zmm29, %zmm6, %zmm30                       #2123.7 c99
..LN4208:
	.loc    1  2124  is_stmt 1
        vpermpd   %zmm9, %zmm15, %zmm3                          #2124.21 c105 stall 2
..LN4209:
	.loc    1  2128  is_stmt 1
        vmovupd   %zmm3, z(%r14){%k1}                           #2128.7 c107
..LN4210:
	.loc    1  2125  is_stmt 1
        vpermpd   %zmm7, %zmm15, %zmm6                          #2125.21 c107
..LN4211:
	.loc    1  2129  is_stmt 1
        vmovupd   %zmm6, z(%r12){%k1}                           #2129.7 c109
..LN4212:
	.loc    1  2126  is_stmt 1
        vpermpd   %zmm5, %zmm15, %zmm8                          #2126.21 c109
..LN4213:
	.loc    1  2130  is_stmt 1
        vmovupd   %zmm8, z(%r8){%k1}                            #2130.7 c113 stall 1
..LN4214:
	.loc    1  2127  is_stmt 1
        vpermpd   %zmm2, %zmm15, %zmm31                         #2127.21 c113
..LN4215:
	.loc    1  2131  is_stmt 1
        vmovupd   %zmm31, z(%rsi){%k1}                          #2131.7 c115
..LN4216:
	.loc    1  2123  is_stmt 1
        vaddpd    %zmm13, %zmm30, %zmm13                        #2123.7 c115
..LN4217:
	.loc    1  2132  is_stmt 1
        vmovupd   %zmm9, z(%r13){%k1}                           #2132.7 c119 stall 1
..LN4218:
	.loc    1  2064  is_stmt 1
        cmpl      %eax, %edx                                    #2064.5 c119
..LN4219:
	.loc    1  2133  is_stmt 1
        vmovupd   %zmm7, z(%r11){%k1}                           #2133.7 c121
..LN4220:
	.loc    1  2134  is_stmt 1
        vmovupd   %zmm5, z(%rdi){%k1}                           #2134.7 c125 stall 1
..LN4221:
	.loc    1  2135  is_stmt 1
        vmovupd   %zmm2, z(%r15){%k1}                           #2135.7 c127
..LN4222:
	.loc    1  2064  is_stmt 1
        jb        ..B12.5       # Prob 82%                      #2064.5 c127
..LN4223:
                                # LOE rcx rbx eax edx r9d r10d zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 k1
..B12.8:                        # Preds ..B12.7
                                # Execution count [6.48e+00]
..LN4224:
        movq      2048(%rsp), %r11                              #[spill] c1
..LN4225:
        movl      2056(%rsp), %r12d                             #[spill] c1
..LN4226:
        movq      2064(%rsp), %rsi                              #[spill] c5 stall 1
..LN4227:
        movq      2072(%rsp), %r8                               #[spill] c5
..LN4228:
        movq      2080(%rsp), %rdi                              #[spill] c9 stall 1
..LN4229:
                                # LOE rsi rdi r8 r11 r9d r10d r12d zmm12 zmm13 zmm15 zmm23
..B12.9:                        # Preds ..B12.8 ..B12.3
                                # Execution count [6.48e+00]
..L544:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.187500
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4230:
	.loc    1  2140  is_stmt 1
..L543:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 4
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4231:
..L542:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4232:
..L541:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.000000
                # ベクトル反復数 1 のループ
                # ベクトル長 4
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4233:
	.loc    1  2138  is_stmt 1
..L540:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4234:
..L539:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 5.335938
                # ベクトル反復数 1 のループ
                # ベクトル長 4
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4235:
	.loc    1  2137  is_stmt 1
..L538:
                # optimization report
                # ループが完全にアンロールされました
                # ループがベクトル化されました
                # CILK PLUS 配列表記ループ
                # ベクトル化のスピードアップ係数 10.671875
                # ベクトル反復数 1 のループ
                # ベクトル長 8
                # メインベクトル型: 64-bits floating point
                # 依存性の解析が無視されました
                # コストモデルの決定が無視されました
..LN4236:
..LN4237:
        vextractf64x4 $0, %zmm13, %ymm0                         #2137.20 c1
..LN4238:
	.loc    1  2138  is_stmt 1
        vextractf64x4 $1, %zmm13, %ymm1                         #2138.21 c1
..LN4239:
	.loc    1  2139  is_stmt 1
        vaddpd    %ymm1, %ymm0, %ymm2                           #2139.27 c3
..LN4240:
	.loc    1  2147  is_stmt 1
        vxorpd    %xmm1, %xmm1, %xmm1                           #2147.16 c3
..LN4241:
	.loc    1  2140  is_stmt 1
        vinsertf64x4 $1, %ymm2, %zmm23, %zmm3                   #2140.18 c9 stall 2
..LN4242:
	.loc    1  2146  is_stmt 1
        vxorpd    %xmm2, %xmm2, %xmm2                           #2146.16 c9
..LN4243:
	.loc    1  2141  is_stmt 1
        vaddpd    %zmm3, %zmm12, %zmm4                          #2141.5 c11
..LN4244:
	.loc    1  2142  is_stmt 1
        vmovups   %zmm4, z(%rsi)                                #2142.31 c17 stall 2
..LN4245:
	.loc    1  2148  is_stmt 1
        vxorpd    %xmm4, %xmm4, %xmm4                           #2148.16 c17
..LN4246:
	.loc    1  2143  is_stmt 1
        vmovsd    z(%rsi), %xmm5                                #2143.24 c23 stall 2
..LN4247:
	.loc    1  2144  is_stmt 1
        vmovsd    8+z(%rsi), %xmm3                              #2144.24 c23
..LN4248:
	.loc    1  2145  is_stmt 1
        vmovsd    16+z(%rsi), %xmm0                             #2145.24 c29 stall 2
..LN4249:
	.loc    1  2149  is_stmt 1
        cmpl      %r12d, %r9d                                   #2149.32 c29
..LN4250:
        jge       ..B12.29      # Prob 0%                       #2149.32 c31
..LN4251:
                                # LOE rsi rdi r8 r11 r9d r10d r12d xmm0 xmm1 xmm2 xmm3 xmm4 xmm5 zmm15 zmm23
..B12.10:                       # Preds ..B12.9
                                # Execution count [0.00e+00]
..LN4252:
	.loc    1  2145  is_stmt 1
        vbroadcastsd %xmm0, %zmm7                               #2145.22 c1
..LN4253:
	.loc    1  2150  is_stmt 1
        movl      %r10d, 1704(%rsp)                             #2150.21[spill] c1
..LN4254:
	.loc    1  2149  is_stmt 1
        movl      $8, %eax                                      #2149.5 c1
..LN4255:
	.loc    1  2150  is_stmt 1
        movq      %r11, 2048(%rsp)                              #2150.21[spill] c3
..LN4256:
	.loc    1  2149  is_stmt 1
        negl      %r9d                                          #2149.5 c3
..LN4257:
	.loc    1  2150  is_stmt 1
        movq      %rsi, 2064(%rsp)                              #2150.21[spill] c5
..LN4258:
	.loc    1  2146  is_stmt 1
        vpxorq    %zmm22, %zmm22, %zmm22                        #2146.16 c5
..LN4259:
	.loc    1  2150  is_stmt 1
        vmovups   %zmm7, 1536(%rsp)                             #2150.21[spill] c7
..LN4260:
	.loc    1  2144  is_stmt 1
        vbroadcastsd %xmm3, %zmm6                               #2144.22 c7
..LN4261:
	.loc    1  2150  is_stmt 1
        vmovups   %zmm6, 1600(%rsp)                             #2150.21[spill] c13 stall 2
..LN4262:
	.loc    1  2149  is_stmt 1
        vmovd     %eax, %xmm3                                   #2149.5 c13
..LN4263:
	.loc    1  2150  is_stmt 1
        movq      %r8, 2072(%rsp)                               #2150.21[spill] c13
..LN4264:
	.loc    1  2146  is_stmt 1
        vmovaps   %zmm22, %zmm14                                #2146.16 c13
..LN4265:
	.loc    1  2150  is_stmt 1
        movq      %rdi, 2080(%rsp)                              #2150.21[spill] c15
..LN4266:
	.loc    1  2149  is_stmt 1
        xorl      %ebx, %ebx                                    #2149.5 c15
..LN4267:
        vpbroadcastd %xmm3, %ymm3                               #2149.5 c15
..LN4268:
        lea       (%r12,%r9), %r13d                             #2149.5 c17
..LN4269:
	.loc    1  2150  is_stmt 1
        movl      %r13d, 1720(%rsp)                             #2150.21[spill] c19
..LN4270:
	.loc    1  2149  is_stmt 1
        lea       63(%r12,%r9), %edx                            #2149.5 c19
..LN4271:
	.loc    1  2150  is_stmt 1
        xorl      %ecx, %ecx                                    #2150.21 c19
..LN4272:
        xorl      %eax, %eax                                    #2150.21 c19
..LN4273:
	.loc    1  2143  is_stmt 1
        vbroadcastsd %xmm5, %zmm21                              #2143.22 c21
..LN4274:
	.loc    1  2157  is_stmt 1
        vmovups   .L_2il0floatpacket.66(%rip), %zmm20           #2157.16 c21
..LN4275:
	.loc    1  2149  is_stmt 1
        shrl      $6, %edx                                      #2149.5 c21
..LN4276:
	.loc    1  2156  is_stmt 1
        vmovups   .L_2il0floatpacket.67(%rip), %zmm19           #2156.59 c27 stall 2
..LN4277:
        vmovups   .L_2il0floatpacket.68(%rip), %zmm18           #2156.21 c27
..LN4278:
        vmovups   .L_2il0floatpacket.69(%rip), %zmm16           #2156.33 c33 stall 2
..LN4279:
	.loc    1  2149  is_stmt 1
        vmovdqu   .L_2il0floatpacket.71(%rip), %ymm0            #2149.5 c33
..LN4280:
                                # LOE eax edx ecx ebx xmm1 xmm2 xmm4 ymm0 ymm3 zmm14 zmm15 zmm16 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.11:                       # Preds ..B12.27 ..B12.10
                                # Execution count [6.48e+00]
..L545:
                # optimization report
                # ピールループ
                # ループが分散されました, チャンク 1
                # %s はベクトル化されませんでした: 内部ループがすでにベクトル化されています。
..LN4281:
        lea       64(%rcx), %edi                                #2149.5 c1
..LN4282:
	.loc    1  2148  is_stmt 1
        vmovaps   %zmm22, %zmm11                                #2148.16 c1
..LN4283:
	.loc    1  2146  is_stmt 1
        vmovaps   %zmm14, %zmm17                                #2146.16 c1
..LN4284:
	.loc    1  2149  is_stmt 1
        cmpl      1720(%rsp), %edi                              #2149.5[spill] c3
..LN4285:
	.loc    1  2147  is_stmt 1
        vmovaps   %zmm22, %zmm7                                 #2147.16 c3
..LN4286:
	.loc    1  2148  is_stmt 1
        vmovaps   %zmm11, %zmm14                                #2148.16 c3
..LN4287:
	.loc    1  2149  is_stmt 1
        cmovg     1720(%rsp), %edi                              #2149.5[spill] c5
..LN4288:
        lea       (%rax,%rdi), %esi                             #2149.5 c7
..LN4289:
        cmpl      $8, %esi                                      #2149.5 c9
..LN4290:
        jl        ..B12.32      # Prob 10%                      #2149.5 c11
..LN4291:
                                # LOE eax edx ecx ebx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.12:                       # Preds ..B12.11
                                # Execution count [6.48e+00]
..LN4292:
	.loc    1  2150  is_stmt 1
        movl      1704(%rsp), %r8d                              #2150.21[spill] c1
..LN4293:
        movq      2048(%rsp), %r11                              #2150.21[spill] c1
..LN4294:
	.loc    1  2149  is_stmt 1
        movl      %esi, %r13d                                   #2149.5 c1
..LN4295:
        xorl      %r10d, %r10d                                  #2149.5 c1
..LN4296:
        andl      $-8, %r13d                                    #2149.5 c3
..LN4297:
	.loc    1  2150  is_stmt 1
        lea       (%rcx,%r8,8), %r9d                            #2150.21 c5
..LN4298:
	.loc    1  2149  is_stmt 1
        movslq    %r13d, %r8                                    #2149.5 c5
..LN4299:
        vbroadcastsd .L_2il0floatpacket.23(%rip), %zmm12        #2149.5 c5
..LN4300:
	.loc    1  2150  is_stmt 1
        movslq    %r9d, %r9                                     #2150.21 c7
..LN4301:
	.loc    1  2149  is_stmt 1
        vmovups   1600(%rsp), %zmm9                             #2149.5[spill] c7
..LN4302:
        vmovups   1536(%rsp), %zmm10                            #2149.5[spill] c11 stall 1
..LN4303:
	.loc    1  2150  is_stmt 1
        lea       (%r11,%r9,4), %r9                             #2150.21 c13
..LN4304:
                                # LOE r8 r9 r10 eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm9 zmm10 zmm11 zmm12 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.13:                       # Preds ..B12.15 ..B12.12
                                # Execution count [3.60e+01]
..L546:
                # optimization report
                # ピールループ
                # ループが分散されました, チャンク 1
                # ループのストリップマイニングが行われました , BY 64
                # ループがベクトル化されました
                # ベクトル化にアライメントされていないメモリー参照が含まれています
                # ベクトル化のスピードアップ係数 3.441406
                # ベクトル・トリップカウント (推定された定数)
                # ベクトル長 8
                # 正規化されたベクトル化のオーバーヘッド 0.500000
                # メインベクトル型: 64-bits floating point
..LN4305:
	.loc    1  2149  is_stmt 1
..LN4306:
	.loc    1  2150  is_stmt 1
        vmovdqu   (%r9,%r10,4), %ymm5                           #2150.21 c1
..LN4307:
        vmovdqu   %ymm5, 1728(%rsp,%r10,4)                      #2150.19 c3
..LN4308:
	.loc    1  2153  is_stmt 1
        vpmovsxdq %ymm5, %zmm8                                  #2153.19 c3
..LN4309:
        kxnorw    %k0, %k0, %k1                                 #2153.19 c3
..LN4310:
        vpxord    %zmm6, %zmm6, %zmm6                           #2153.19 c3
..LN4311:
        kxnorw    %k0, %k0, %k2                                 #2153.19 c3
..LN4312:
        vpsllq    $3, %zmm8, %zmm24                             #2153.19 c5
..LN4313:
        vpxord    %zmm13, %zmm13, %zmm13                        #2153.19 c5
..LN4314:
        kxnorw    %k0, %k0, %k3                                 #2153.19 c5
..LN4315:
        vpxord    %zmm25, %zmm25, %zmm25                        #2153.19 c7
..LN4316:
	.loc    1  2156  is_stmt 1
        vmovaps   %zmm12, %zmm29                                #2156.52 c7
..LN4317:
	.loc    1  2153  is_stmt 1
        vgatherqpd 16+z(,%zmm24,8), %zmm6{%k1}                  #2153.19 c7
..LN4318:
        vgatherqpd 8+z(,%zmm24,8), %zmm13{%k2}                  #2153.19 c7
..LN4319:
        vsubpd    %zmm10, %zmm6, %zmm8                          #2153.29 c13 stall 2
..LN4320:
	.loc    1  2152  is_stmt 1
        vsubpd    %zmm9, %zmm13, %zmm6                          #2152.29 c13
..LN4321:
	.loc    1  2153  is_stmt 1
        vgatherqpd z(,%zmm24,8), %zmm25{%k3}                    #2153.19 c13
..LN4322:
	.loc    1  2151  is_stmt 1
        vsubpd    %zmm21, %zmm25, %zmm5                         #2151.29 c19 stall 2
..LN4323:
	.loc    1  2154  is_stmt 1
        vmulpd    %zmm6, %zmm6, %zmm31                          #2154.35 c19
..LN4324:
        vfmadd231pd %zmm5, %zmm5, %zmm31                        #2154.35 c25 stall 2
..LN4325:
	.loc    1  2156  is_stmt 1
        vmovaps   %zmm16, %zmm13                                #2156.33 c25
..LN4326:
	.loc    1  2154  is_stmt 1
        vfmadd231pd %zmm8, %zmm8, %zmm31                        #2154.45 c31 stall 2
..LN4327:
	.loc    1  2155  is_stmt 1
        vmulpd    %zmm31, %zmm31, %zmm26                        #2155.24 c37 stall 2
..LN4328:
        vmulpd    %zmm26, %zmm31, %zmm27                        #2155.29 c43 stall 2
..LN4329:
	.loc    1  2156  is_stmt 1
        vmulpd    %zmm27, %zmm31, %zmm28                        #2156.47 c49 stall 2
..LN4330:
        vfmsub231pd %zmm27, %zmm18, %zmm13                      #2156.33 c49
..LN4331:
        vmulpd    %zmm28, %zmm27, %zmm24                        #2156.52 c55 stall 2
..LN4332:
        vrcp28pd  {sae}, %zmm24, %zmm30                         #2156.52 c61 stall 2
..LN4333:
        vfnmadd231pd {rn-sae}, %zmm24, %zmm30, %zmm29           #2156.52 c69 stall 3
..LN4334:
        vfmadd213pd {rn-sae}, %zmm30, %zmm29, %zmm30            #2156.52 c75 stall 2
..LN4335:
        vcmppd    $8, %zmm22, %zmm30, %k1                       #2156.52 c81 stall 2
..LN4336:
        vmulpd    %zmm30, %zmm13, %zmm25                        #2156.52 c81
..LN4337:
        kortestw  %k1, %k1                                      #2156.52 c83
..LN4338:
        je        ..B12.15      # Prob 25%                      #2156.52 c85
..LN4339:
                                # LOE r8 r9 r10 eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm31 k1
..B12.14:                       # Preds ..B12.13
                                # Execution count [2.70e+01]
..LN4340:
        vdivpd    %zmm24, %zmm13, %zmm25{%k1}                   #2156.52 c3 stall 1
..LN4341:
                                # LOE r8 r9 r10 eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm25 zmm31
..B12.15:                       # Preds ..B12.14 ..B12.13
                                # Execution count [3.60e+01]
..LN4342:
	.loc    1  2157  is_stmt 1
        vcmppd    $10, %zmm20, %zmm31, %k1                      #2157.16 c1
..LN4343:
	.loc    1  2156  is_stmt 1
        vmulpd    %zmm25, %zmm19, %zmm13{%k1}{z}                #2156.59 c75 stall 36
..LN4344:
	.loc    1  2158  is_stmt 1
        vfmadd231pd %zmm5, %zmm13, %zmm17                       #2158.7 c81 stall 2
..LN4345:
        vmulpd    %zmm13, %zmm5, %zmm5                          #2158.19 c81
..LN4346:
	.loc    1  2159  is_stmt 1
        vfmadd231pd %zmm6, %zmm13, %zmm7                        #2159.7 c87 stall 2
..LN4347:
        vmulpd    %zmm13, %zmm6, %zmm6                          #2159.19 c87
..LN4348:
	.loc    1  2160  is_stmt 1
        vmulpd    %zmm13, %zmm8, %zmm25                         #2160.19 c93 stall 2
..LN4349:
        vfmadd231pd %zmm8, %zmm13, %zmm11                       #2160.7 c93
..LN4350:
	.loc    1  2161  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm5, %zmm8 #2161.7 c93
..LN4351:
        vmovupd   %zmm8, (%rsp,%r10,8)                          #2161.7 c95
..LN4352:
	.loc    1  2162  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm6, %zmm24 #2162.7 c95
..LN4353:
        vmovupd   %zmm24, 512(%rsp,%r10,8)                      #2162.7 c97
..LN4354:
	.loc    1  2163  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm25, %zmm26 #2163.7 c101 stall 1
..LN4355:
        vmovupd   %zmm26, 1024(%rsp,%r10,8)                     #2163.7 c103
..LN4356:
	.loc    1  2149  is_stmt 1
        addq      $8, %r10                                      #2149.5 c103
..LN4357:
        cmpq      %r8, %r10                                     #2149.5 c105
..LN4358:
        jb        ..B12.13      # Prob 82%                      #2149.5 c107
..LN4359:
                                # LOE r8 r9 r10 eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm9 zmm10 zmm11 zmm12 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.17:                       # Preds ..B12.15 ..B12.32
                                # Execution count [7.20e+00]
..LN4360:
        lea       1(%r13), %r8d                                 #2149.5 c1
..LN4361:
        cmpl      %esi, %r8d                                    #2149.5 c3
..LN4362:
        ja        ..B12.23      # Prob 50%                      #2149.5 c5
..LN4363:
                                # LOE eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.18:                       # Preds ..B12.17
                                # Execution count [6.48e+00]
..LN4364:
	.loc    1  2150  is_stmt 1
        movslq    %r13d, %r14                                   #2150.19 c1
..LN4365:
	.loc    1  2149  is_stmt 1
        movl      %ebx, 2088(%rsp)                              #2149.5[spill] c1
..LN4366:
        negl      %r13d                                         #2149.5 c1
..LN4367:
        vmovdqa   %ymm0, %ymm8                                  #2149.5 c1
..LN4368:
        vbroadcastsd .L_2il0floatpacket.23(%rip), %zmm24        #2149.5 c1
..LN4369:
        addl      %esi, %r13d                                   #2149.5 c3
..LN4370:
	.loc    1  2150  is_stmt 1
        movl      $255, %r8d                                    #2150.21 c3
..LN4371:
        lea       1728(%rsp,%r14,4), %r9                        # c3
..LN4372:
	.loc    1  2149  is_stmt 1
        vmovd     %r13d, %xmm6                                  #2149.5 c5
..LN4373:
	.loc    1  2150  is_stmt 1
        movl      1704(%rsp), %r13d                             #2150.21[spill] c5
..LN4374:
        kmovw     %r8d, %k1                                     #2150.21 c5
..LN4375:
	.loc    1  2149  is_stmt 1
        xorl      %r8d, %r8d                                    #2149.5 c5
..LN4376:
        vpbroadcastd %xmm6, %ymm5                               #2149.5 c7
..LN4377:
        lea       1024(%rsp,%r14,8), %r10                       # c7
..LN4378:
	.loc    1  2150  is_stmt 1
        lea       (%rcx,%r13,8), %r15d                          #2150.21 c9
..LN4379:
        movq      2048(%rsp), %r13                              #2150.21[spill] c9
..LN4380:
        movslq    %r15d, %r15                                   #2150.21 c11
..LN4381:
        lea       512(%rsp,%r14,8), %r11                        # c11
..LN4382:
        addq      %r14, %r15                                    #2150.21 c13
..LN4383:
        lea       (%rsp,%r14,8), %r12                           # c13
..LN4384:
	.loc    1  2149  is_stmt 1
        vmovups   1600(%rsp), %zmm25                            #2149.5[spill] c13
..LN4385:
	.loc    1  2150  is_stmt 1
        lea       (%r13,%r15,4), %r13                           #2150.21 c15
..LN4386:
        movq      %r13, 1712(%rsp)                              #2150.21[spill] c17
..LN4387:
	.loc    1  2149  is_stmt 1
        movslq    %esi, %r13                                    #2149.5 c17
..LN4388:
        xorl      %r15d, %r15d                                  #2149.5 c17
..LN4389:
        subq      %r14, %r13                                    #2149.5 c19
..LN4390:
        xorl      %r14d, %r14d                                  #2149.5 c19
..LN4391:
        vmovups   1536(%rsp), %zmm26                            #2149.5[spill] c19
..LN4392:
        movq      1712(%rsp), %rbx                              #2149.5[spill] c19
..LN4393:
                                # LOE rbx r8 r9 r10 r11 r12 r13 r14 r15 eax edx ecx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 ymm8 zmm5 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 k1
..B12.19:                       # Preds ..B12.21 ..B12.18
                                # Execution count [3.60e+01]
..L547:
                # optimization report
                # ピールループ
                # ループが分散されました, チャンク 1
                # ループのストリップマイニングが行われました , BY 64
                # ループがベクトル化されました
                # ベクトル化の剰余ループ
                # マスク付きベクトル化
                # ベクトル化にアライメントされていないメモリー参照が含まれています
                # ベクトル化のスピードアップ係数 1.776367
                # ベクトル・トリップカウント (推定された定数)
                # ベクトル長 8
                # 正規化されたベクトル化のオーバーヘッド 0.406250
                # メインベクトル型: 64-bits floating point
..LN4394:
	.loc    1  2153  is_stmt 1
        vpxord    %zmm6, %zmm6, %zmm6                           #2153.19 c1
..LN4395:
	.loc    1  2150  is_stmt 1
        vpcmpgtd  %zmm8, %zmm5, %k5{%k1}                        #2150.21 c3
..LN4396:
	.loc    1  2153  is_stmt 1
        vpxord    %zmm10, %zmm10, %zmm10                        #2153.19 c3
..LN4397:
	.loc    1  2150  is_stmt 1
        vmovdqu32 (%r15,%rbx), %zmm13{%k5}{z}                   #2150.21 c5
..LN4398:
        vmovdqu32 %zmm13, (%r15,%r9){%k5}                       #2150.19 c11 stall 2
..LN4399:
	.loc    1  2153  is_stmt 1
        kmovw     %k5, %k3                                      #2153.19 c11
..LN4400:
        kmovw     %k5, %k2                                      #2153.19 c11
..LN4401:
        vpxord    %zmm28, %zmm28, %zmm28                        #2153.19 c11
..LN4402:
        kmovw     %k5, %k4                                      #2153.19 c13
..LN4403:
	.loc    1  2149  is_stmt 1
        vmovdqu32 (%r15,%r9), %zmm12{%k5}{z}                    #2149.5 c17 stall 1
..LN4404:
	.loc    1  2153  is_stmt 1
        vpmovsxdq %ymm12, %zmm9                                 #2153.19 c25
..LN4405:
        vpsllq    $3, %zmm9, %zmm27                             #2153.19 c27
..LN4406:
        vgatherqpd z(,%zmm27,8), %zmm28{%k4}                    #2153.19 c29
..LN4407:
        vgatherqpd 8+z(,%zmm27,8), %zmm6{%k3}                   #2153.19 c29
..LN4408:
	.loc    1  2152  is_stmt 1
        vsubpd    %zmm25, %zmm6, %zmm12                         #2152.29 c35 stall 2
..LN4409:
	.loc    1  2156  is_stmt 1
        vmovaps   %zmm16, %zmm6                                 #2156.33 c35
..LN4410:
	.loc    1  2153  is_stmt 1
        vgatherqpd 16+z(,%zmm27,8), %zmm10{%k2}                 #2153.19 c35
..LN4411:
	.loc    1  2156  is_stmt 1
        vmovaps   %zmm24, %zmm27                                #2156.52 c37
..LN4412:
	.loc    1  2153  is_stmt 1
        vsubpd    %zmm26, %zmm10, %zmm13                        #2153.29 c41 stall 1
..LN4413:
	.loc    1  2151  is_stmt 1
        vsubpd    %zmm21, %zmm28, %zmm10                        #2151.29 c41
..LN4414:
	.loc    1  2154  is_stmt 1
        vmulpd    %zmm12, %zmm12, %zmm9                         #2154.35 c47 stall 2
..LN4415:
        vfmadd231pd %zmm10, %zmm10, %zmm9                       #2154.35 c53 stall 2
..LN4416:
        vfmadd231pd %zmm13, %zmm13, %zmm9                       #2154.45 c59 stall 2
..LN4417:
	.loc    1  2155  is_stmt 1
        vmulpd    %zmm9, %zmm9, %zmm29                          #2155.24 c65 stall 2
..LN4418:
        vmulpd    %zmm29, %zmm9, %zmm30                         #2155.29 c71 stall 2
..LN4419:
	.loc    1  2156  is_stmt 1
        vmulpd    %zmm30, %zmm9, %zmm31                         #2156.47 c77 stall 2
..LN4420:
        vfmsub231pd %zmm30, %zmm18, %zmm6                       #2156.33 c77
..LN4421:
        vmulpd    %zmm31, %zmm30, %zmm29                        #2156.52 c83 stall 2
..LN4422:
        vrcp28pd  {sae}, %zmm29, %zmm28                         #2156.52 c89 stall 2
..LN4423:
        vfnmadd231pd {rn-sae}, %zmm29, %zmm28, %zmm27           #2156.52 c97 stall 3
..LN4424:
        vfmadd213pd {rn-sae}, %zmm28, %zmm27, %zmm28            #2156.52 c103 stall 2
..LN4425:
        vcmppd    $8, %zmm22, %zmm28, %k2                       #2156.52 c109 stall 2
..LN4426:
        vmulpd    %zmm28, %zmm6, %zmm27                         #2156.52 c109
..LN4427:
        kortestw  %k2, %k2                                      #2156.52 c111
..LN4428:
        je        ..B12.21      # Prob 25%                      #2156.52 c113
..LN4429:
                                # LOE rbx r8 r9 r10 r11 r12 r13 r14 r15 eax edx ecx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 ymm8 zmm5 zmm6 zmm7 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm29 k1 k2 k5
..B12.20:                       # Preds ..B12.19
                                # Execution count [2.70e+01]
..LN4430:
        vdivpd    %zmm29, %zmm6, %zmm27{%k2}                    #2156.52 c3 stall 1
..LN4431:
                                # LOE rbx r8 r9 r10 r11 r12 r13 r14 r15 eax edx ecx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 ymm8 zmm5 zmm7 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 k1 k5
..B12.21:                       # Preds ..B12.20 ..B12.19
                                # Execution count [3.60e+01]
..LN4432:
	.loc    1  2157  is_stmt 1
        vcmppd    $10, %zmm20, %zmm9, %k2{%k5}                  #2157.16 c1
..LN4433:
	.loc    1  2149  is_stmt 1
        addq      $8, %r8                                       #2149.5 c1
..LN4434:
        addq      $32, %r15                                     #2149.5 c1
..LN4435:
        vpaddd    %ymm3, %ymm8, %ymm8                           #2149.5 c1
..LN4436:
	.loc    1  2156  is_stmt 1
        vmulpd    %zmm27, %zmm19, %zmm6{%k2}{z}                 #2156.59 c75 stall 36
..LN4437:
	.loc    1  2158  is_stmt 1
        vmulpd    %zmm6, %zmm10, %zmm9                          #2158.19 c81 stall 2
..LN4438:
	.loc    1  2159  is_stmt 1
        vmulpd    %zmm6, %zmm12, %zmm10                         #2159.19 c81
..LN4439:
	.loc    1  2160  is_stmt 1
        vmulpd    %zmm6, %zmm13, %zmm27                         #2160.19 c87 stall 2
..LN4440:
	.loc    1  2161  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm9, %zmm13 #2161.7 c87
..LN4441:
        vmovupd   %zmm13, (%r14,%r12){%k5}                      #2161.7 c89
..LN4442:
	.loc    1  2162  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm10, %zmm12 #2162.7 c89
..LN4443:
        vmovupd   %zmm12, (%r14,%r11){%k5}                      #2162.7 c91
..LN4444:
	.loc    1  2158  is_stmt 1
        vaddpd    %zmm9, %zmm17, %zmm17{%k5}                    #2158.7 c91
..LN4445:
	.loc    1  2159  is_stmt 1
        vaddpd    %zmm10, %zmm7, %zmm7{%k5}                     #2159.7 c93
..LN4446:
	.loc    1  2163  is_stmt 1
        vpxorq    .L_2il0floatpacket.70(%rip){1to8}, %zmm27, %zmm28 #2163.7 c95
..LN4447:
        vmovupd   %zmm28, (%r14,%r10){%k5}                      #2163.7 c97
..LN4448:
	.loc    1  2149  is_stmt 1
        addq      $64, %r14                                     #2149.5 c97
..LN4449:
	.loc    1  2160  is_stmt 1
        vaddpd    %zmm27, %zmm11, %zmm11{%k5}                   #2160.7 c97
..LN4450:
	.loc    1  2149  is_stmt 1
        cmpq      %r13, %r8                                     #2149.5 c97
..LN4451:
        jb        ..B12.19      # Prob 82%                      #2149.5 c99
..LN4452:
                                # LOE rbx r8 r9 r10 r11 r12 r13 r14 r15 eax edx ecx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 ymm8 zmm5 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 k1
..B12.22:                       # Preds ..B12.21
                                # Execution count [6.48e+00]
..LN4453:
        movl      2088(%rsp), %ebx                              #[spill] c1
..LN4454:
                                # LOE eax edx ecx ebx esi edi xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.23:                       # Preds ..B12.22 ..B12.17
                                # Execution count [7.20e+00]
..LN4455:
	.loc    1  2147  is_stmt 1
        vextractf64x4 $1, %zmm7, %ymm8                          #2147.16 c1
..LN4456:
	.loc    1  2149  is_stmt 1
        xorl      %r8d, %r8d                                    #2149.5 c1
..LN4457:
	.loc    1  2148  is_stmt 1
        vextractf64x4 $1, %zmm11, %ymm12                        #2148.16 c3
..LN4458:
	.loc    1  2147  is_stmt 1
        vaddpd    %ymm7, %ymm8, %ymm5                           #2147.16 c5
..LN4459:
	.loc    1  2148  is_stmt 1
        vaddpd    %ymm11, %ymm12, %ymm13                        #2148.16 c5
..LN4460:
	.loc    1  2147  is_stmt 1
        valignq   $3, %zmm5, %zmm5, %zmm28                      #2147.16 c13
..LN4461:
        valignq   $2, %zmm5, %zmm5, %zmm29                      #2147.16 c13
..LN4462:
        valignq   $1, %zmm5, %zmm5, %zmm30                      #2147.16 c15
..LN4463:
	.loc    1  2148  is_stmt 1
        valignq   $3, %zmm13, %zmm13, %zmm24                    #2148.16 c15
..LN4464:
        valignq   $2, %zmm13, %zmm13, %zmm25                    #2148.16 c17
..LN4465:
        valignq   $1, %zmm13, %zmm13, %zmm26                    #2148.16 c17
..LN4466:
	.loc    1  2147  is_stmt 1
        vaddsd    %xmm30, %xmm29, %xmm6                         #2147.16 c27
..LN4467:
        vaddsd    %xmm5, %xmm28, %xmm5                          #2147.16 c27
..LN4468:
        vaddsd    %xmm5, %xmm6, %xmm31                          #2147.16 c33 stall 2
..LN4469:
	.loc    1  2146  is_stmt 1
        vextractf64x4 $1, %zmm17, %ymm6                         #2146.16 c33
..LN4470:
        vmovaps   %zmm17, %zmm8                                 #2146.16 c35
..LN4471:
	.loc    1  2148  is_stmt 1
        vaddsd    %xmm26, %xmm25, %xmm10                        #2148.16 c37
..LN4472:
        vaddsd    %xmm13, %xmm24, %xmm9                         #2148.16 c39
..LN4473:
        vaddsd    %xmm9, %xmm10, %xmm27                         #2148.16 c43 stall 1
..LN4474:
	.loc    1  2146  is_stmt 1
        vaddpd    %ymm8, %ymm6, %ymm9                           #2146.16 c45
..LN4475:
	.loc    1  2148  is_stmt 1
        vaddsd    %xmm4, %xmm27, %xmm4                          #2148.16 c49 stall 1
..LN4476:
	.loc    1  2146  is_stmt 1
        valignq   $3, %zmm9, %zmm9, %zmm17                      #2146.16 c53
..LN4477:
        valignq   $2, %zmm9, %zmm9, %zmm7                       #2146.16 c55
..LN4478:
        valignq   $1, %zmm9, %zmm9, %zmm24                      #2146.16 c55
..LN4479:
        vaddsd    %xmm24, %xmm7, %xmm7                          #2146.16 c61
..LN4480:
        vaddsd    %xmm9, %xmm17, %xmm6                          #2146.16 c61
..LN4481:
        vaddsd    %xmm6, %xmm7, %xmm5                           #2146.16 c67 stall 2
..LN4482:
	.loc    1  2147  is_stmt 1
        vaddsd    %xmm1, %xmm31, %xmm1                          #2147.16 c67
..LN4483:
	.loc    1  2146  is_stmt 1
        vaddsd    %xmm2, %xmm5, %xmm2                           #2146.16 c73 stall 2
..LN4484:
	.loc    1  2149  is_stmt 1
        testl     %esi, %esi                                    #2149.5 c73
..LN4485:
        je        ..B12.27      # Prob 10%                      #2149.5 c75
..LN4486:
                                # LOE r8 eax edx ecx ebx edi xmm1 xmm2 xmm4 ymm0 ymm3 zmm14 zmm15 zmm16 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.24:                       # Preds ..B12.23
                                # Execution count [6.48e+00]
..LN4487:
        movslq    %ebx, %rsi                                    #2149.5 c1
..LN4488:
        movslq    %edi, %rdi                                    #2149.5 c1
..LN4489:
        shlq      $6, %rsi                                      #2149.5 c3
..LN4490:
        negq      %rsi                                          #2149.5 c5
..LN4491:
        addq      %rdi, %rsi                                    #2149.5 c7
        .align    16,0x90
..LN4492:
                                # LOE rsi r8 eax edx ecx ebx xmm1 xmm2 xmm4 ymm0 ymm3 zmm14 zmm15 zmm16 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.25:                       # Preds ..B12.25 ..B12.24
                                # Execution count [3.60e+01]
..L548:
                # optimization report
                # ピールループ
                # ループが分散されました, チャンク 2
                # ループのストリップマイニングが行われました , BY 64
                # %s はベクトル化されませんでした: ベクトル化は可能ですが非効率です。オーバーライドするには vector always ディレクティブまたは -vec-threshold0 を使用してください。
                # ベクトル化のスピードアップ係数 0.590820
                # ベクトル・トリップカウント (推定された定数)
                # ベクトル長 2
..LN4493:
	.loc    1  2161  is_stmt 1
        movslq    1728(%rsp,%r8,4), %rdi                        #2161.7 c1
..LN4494:
        shlq      $6, %rdi                                      #2161.7 c5 stall 1
..LN4495:
        vmovsd    32+z(%rdi), %xmm5                             #2161.7 c7
..LN4496:
	.loc    1  2162  is_stmt 1
        vmovsd    40+z(%rdi), %xmm7                             #2162.7 c7
..LN4497:
	.loc    1  2163  is_stmt 1
        vmovsd    48+z(%rdi), %xmm9                             #2163.7 c13 stall 2
..LN4498:
	.loc    1  2161  is_stmt 1
        vaddsd    (%rsp,%r8,8), %xmm5, %xmm6                    #2161.7 c13
..LN4499:
        vmovsd    %xmm6, 32+z(%rdi)                             #2161.7 c19 stall 2
..LN4500:
	.loc    1  2162  is_stmt 1
        vaddsd    512(%rsp,%r8,8), %xmm7, %xmm8                 #2162.7 c19
..LN4501:
        vmovsd    %xmm8, 40+z(%rdi)                             #2162.7 c25 stall 2
..LN4502:
	.loc    1  2163  is_stmt 1
        vaddsd    1024(%rsp,%r8,8), %xmm9, %xmm10               #2163.7 c25
..LN4503:
        vmovsd    %xmm10, 48+z(%rdi)                            #2163.7 c31 stall 2
..LN4504:
	.loc    1  2149  is_stmt 1
        addq      $1, %r8                                       #2149.5 c31
..LN4505:
        cmpq      %rsi, %r8                                     #2149.5 c33
..LN4506:
        jb        ..B12.25      # Prob 82%                      #2149.5 c35
..LN4507:
                                # LOE rsi r8 eax edx ecx ebx xmm1 xmm2 xmm4 ymm0 ymm3 zmm14 zmm15 zmm16 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.27:                       # Preds ..B12.25 ..B12.23
                                # Execution count [3.60e+01]
..LN4508:
        addl      $1, %ebx                                      #2149.5 c1
..LN4509:
        addl      $64, %ecx                                     #2149.5 c1
..LN4510:
        addl      $-64, %eax                                    #2149.5 c3
..LN4511:
        cmpl      %edx, %ebx                                    #2149.5 c3
..LN4512:
        jb        ..B12.11      # Prob 82%                      #2149.5 c5
..LN4513:
                                # LOE eax edx ecx ebx xmm1 xmm2 xmm4 ymm0 ymm3 zmm14 zmm15 zmm16 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..B12.28:                       # Preds ..B12.27
                                # Execution count [6.48e+00]
..LN4514:
        movq      2064(%rsp), %rsi                              #[spill] c1
..LN4515:
        movq      2072(%rsp), %r8                               #[spill] c1
..LN4516:
        movq      2080(%rsp), %rdi                              #[spill] c5 stall 1
..LN4517:
                                # LOE rsi rdi r8 xmm1 xmm2 xmm4 zmm15 zmm23
..B12.29:                       # Preds ..B12.28 ..B12.9
                                # Execution count [7.20e+00]
..LN4518:
	.loc    1  2165  is_stmt 1
        vaddsd    32+z(%rsi), %xmm2, %xmm0                      #2165.5 c1
..LN4519:
        vmovsd    %xmm0, 32+z(%rsi)                             #2165.5 c7 stall 2
..LN4520:
	.loc    1  2166  is_stmt 1
        vaddsd    40+z(%rsi), %xmm1, %xmm1                      #2166.5 c7
..LN4521:
        vmovsd    %xmm1, 40+z(%rsi)                             #2166.5 c13 stall 2
..LN4522:
	.loc    1  2167  is_stmt 1
        vaddsd    48+z(%rsi), %xmm4, %xmm2                      #2167.5 c13
..LN4523:
        vmovsd    %xmm2, 48+z(%rsi)                             #2167.5 c19 stall 2
..LN4524:
	.loc    1  2058  is_stmt 1
        addq      $1, %r8                                       #2058.3 c19
..LN4525:
        addq      $64, %rsi                                     #2058.3 c19
..LN4526:
        cmpq      %rdi, %r8                                     #2058.3 c21
..LN4527:
        jb        ..B12.3       # Prob 87%                      #2058.3 c23
..LN4528:
                                # LOE rsi rdi r8 zmm15 zmm23
..B12.30:                       # Preds ..B12.29
                                # Execution count [9.00e-01]
..LN4529:
        movq      1664(%rsp), %r12                              #[spill] c1
	.cfi_restore 12
..LN4530:
        movq      1672(%rsp), %r13                              #[spill] c1
	.cfi_restore 13
..LN4531:
        movq      1680(%rsp), %r14                              #[spill] c5 stall 1
	.cfi_restore 14
..LN4532:
        movq      1688(%rsp), %r15                              #[spill] c5
	.cfi_restore 15
..LN4533:
        movq      1696(%rsp), %rbx                              #[spill] c9 stall 1
	.cfi_restore 3
..LN4534:
                                # LOE rbx r12 r13 r14 r15
..B12.31:                       # Preds ..B12.30 ..B12.1
                                # Execution count [1.00e+00]
..LN4535:
	.loc    1  2169  epilogue_begin  is_stmt 1
        movq      %rbp, %rsp                                    #2169.1 c5
..LN4536:
        popq      %rbp                                          #2169.1
	.cfi_def_cfa 7, 8
	.cfi_restore 6
..LN4537:
        ret                                                     #2169.1
	.cfi_def_cfa 6, 16
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xfe, 0xff, 0xff, 0x22
	.cfi_offset 6, -16
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xfe, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xfe, 0xff, 0xff, 0x22
..LN4538:
                                # LOE
..B12.32:                       # Preds ..B12.11
                                # Execution count [6.48e-01]: Infreq
..LN4539:
	.loc    1  2149  is_stmt 1
        xorl      %r13d, %r13d                                  #2149.5 c1
..LN4540:
        jmp       ..B12.17      # Prob 100%                     #2149.5 c1
        .align    16,0x90
..LN4541:
                                # LOE eax edx ecx ebx esi edi r13d xmm1 xmm2 xmm4 ymm0 ymm3 zmm7 zmm11 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23
..LN4542:
	.cfi_endproc
# mark_end;
	.type	_Z21force_sorted_z_intrinv,@function
	.size	_Z21force_sorted_z_intrinv,.-_Z21force_sorted_z_intrinv
..LN_Z21force_sorted_z_intrinv.4543:
.LN_Z21force_sorted_z_intrinv:
	.data
# -- End  _Z21force_sorted_z_intrinv
