# -- Begin  force_intrin_v6_noswp()
	.text
# mark_begin;
# Threads 2
        .align    16,0x90
	.globl force_intrin_v6_noswp()
# --- force_intrin_v6_noswp()
force_intrin_v6_noswp():
..B34.1:                        # Preds ..B34.0
                                # Execution count [1.00e+00]
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z21force_intrin_v6_noswpv.996:
..L997:
                                                        #1962.29
..LN8441:
	.loc    1  1962  prologue_end  is_stmt 1
..LN8442:
	.loc    1  1968  is_stmt 1
        movslq    particle_number(%rip), %r8                    #1968.23 c1
..LN8443:
	.loc    1  1963  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.50(%rip), %zmm14        #1963.23 c1
..LN8444:
	.loc    1  1967  is_stmt 1
        vpxord    %zmm11, %zmm11, %zmm11                        #1967.23 c1
..LN8445:
	.loc    1  1971  is_stmt 1
        xorl      %edi, %edi                                    #1971.14 c1
..LN8446:
        xorl      %esi, %esi                                    #1971.14 c3
..LN8447:
	.loc    1  1964  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.51(%rip), %zmm13        #1964.23 c5
..LN8448:
	.loc    1  1965  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.46(%rip), %zmm7         #1965.23 c7
..LN8449:
	.loc    1  1966  is_stmt 1
        vbroadcastsd .L_2il0floatpacket.21(%rip), %zmm12        #1966.23 c11 stall 1
..LN8450:
	.loc    1  1969  is_stmt 1
        vpbroadcastq .L_2il0floatpacket.52(%rip), %zmm16        #1969.23 c13
..LN8451:
	.loc    1  1971  is_stmt 1
        testq     %r8, %r8                                      #1971.23 c13
..LN8452:
        jle       ..B34.9       # Prob 10%                      #1971.23 c15
..LN8453:
                                # LOE rbx rbp rsi rdi r8 r12 r13 r14 r15 zmm7 zmm11 zmm12 zmm13 zmm14 zmm16
..B34.2:                        # Preds ..B34.1
                                # Execution count [9.06e-01]
..LN8454:
	.loc    1  1976  is_stmt 1
        vmovaps   %zmm11, %zmm4                                 #1976.17 c1
..LN8455:
	.loc    1  1980  is_stmt 1
        movq      number_of_partners(%rip), %rcx                #1980.21 c1
..LN8456:
	.loc    1  1981  is_stmt 1
        movq      pointer(%rip), %r9                            #1981.21 c1
..LN8457:
	.loc    1  1982  is_stmt 1
        movq      sorted_list(%rip), %rdx                       #1982.28 c5 stall 1
..LN8458:
	.loc    1  1984  is_stmt 1
        vmovups   .L_2il0floatpacket.53(%rip), %zmm17           #1984.19 c5
..LN8459:
                                # LOE rdx rcx rbx rbp rsi rdi r8 r9 r12 r13 r14 r15 zmm4 zmm7 zmm11 zmm12 zmm13 zmm14 zmm16 zmm17
..B34.3:                        # Preds ..B34.7 ..B34.2
                                # Execution count [3.57e+05]
..L999:
                # optimization report
                # ユーザー定義のベクトル組込み関数を含むループ
                # 外部 %s は自動ベクトル化されませんでした: SIMD ディレクティブの使用を検討してください。
..LN8460:
	.loc    1  1971  is_stmt 1
..LN8461:
	.loc    1  1980  is_stmt 1
        movl      (%rcx,%rdi,4), %r10d                          #1980.21 c1
..LN8462:
	.loc    1  1981  is_stmt 1
        movslq    (%r9,%rdi,4), %rax                            #1981.21 c1
..LN8463:
	.loc    1  1978  is_stmt 1
        vmovaps   %zmm11, %zmm10                                #1978.17 c1
..LN8464:
	.loc    1  1976  is_stmt 1
        vmovaps   %zmm4, %zmm18                                 #1976.17 c1
..LN8465:
	.loc    1  1986  is_stmt 1
        xorl      %r11d, %r11d                                  #1986.25 c1
..LN8466:
	.loc    1  1977  is_stmt 1
        vmovaps   %zmm11, %zmm19                                #1977.17 c3
..LN8467:
	.loc    1  1978  is_stmt 1
        vmovaps   %zmm10, %zmm4                                 #1978.17 c3
..LN8468:
	.loc    1  1972  is_stmt 1
        vbroadcastsd z(%rsi), %zmm9                             #1972.23 c5
..LN8469:
	.loc    1  1973  is_stmt 1
        vbroadcastsd 8+z(%rsi), %zmm8                           #1973.23 c5
..LN8470:
	.loc    1  1984  is_stmt 1
        vmovaps   %zmm17, %zmm15                                #1984.19 c5
..LN8471:
	.loc    1  1974  is_stmt 1
        vbroadcastsd 16+z(%rsi), %zmm6                          #1974.23 c11 stall 2
..LN8472:
	.loc    1  1982  is_stmt 1
        lea       (%rdx,%rax,4), %rax                           #1982.28 c11
..LN8473:
	.loc    1  1986  is_stmt 1
        testl     %r10d, %r10d                                  #1986.25 c11
..LN8474:
	.loc    1  1983  is_stmt 1
        movslq    %r10d, %r10                                   #1983.22 c11
..LN8475:
        vpbroadcastq %r10, %zmm5                                #1983.22 c13
..LN8476:
	.loc    1  1986  is_stmt 1
        jle       ..B34.7       # Prob 10%                      #1986.25 c13
..LN8477:
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r12 r13 r14 r15 r10d r11d zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19
..B34.5:                        # Preds ..B34.3 ..B34.5
                                # Execution count [1.79e+06]
..L1000:
                # optimization report
                # ユーザー定義のベクトル組込み関数を含むループ
                # %s はベクトル化されませんでした: ベクトル依存関係がベクトル化を妨げています。%s
                # ベクトル・トリップカウント (推定された定数)
..LN8478:
..LN8479:
	.loc    1  1987  is_stmt 1
        vlddqu    (%rax), %ymm0                                 #1987.45 c1
..LN8480:
	.loc    1  1989  is_stmt 1
        vpcmpgtq  %zmm15, %zmm5, %k4                            #1989.27 c1
..LN8481:
	.loc    1  1990  is_stmt 1
        vmovaps   %zmm11, %zmm20                                #1990.27 c1
..LN8482:
	.loc    1  1986  is_stmt 1
        addl      $8, %r11d                                     #1986.29 c1
..LN8483:
	.loc    1  1988  is_stmt 1
        addq      $32, %rax                                     #1988.7 c1
..LN8484:
	.loc    1  1987  is_stmt 1
        vpslld    $3, %ymm0, %ymm1                              #1987.27 c3
..LN8485:
	.loc    1  1990  is_stmt 1
        kmovw     %k4, %k1                                      #1990.27 c3
..LN8486:
	.loc    1  1991  is_stmt 1
        vmovaps   %zmm11, %zmm21                                #1991.27 c3
..LN8487:
	.loc    1  1992  is_stmt 1
        vmovaps   %zmm11, %zmm22                                #1992.27 c3
..LN8488:
	.loc    1  1991  is_stmt 1
        kmovw     %k4, %k2                                      #1991.27 c5
..LN8489:
	.loc    1  1992  is_stmt 1
        kmovw     %k4, %k3                                      #1992.27 c5
..LN8490:
	.loc    1  2004  is_stmt 1
        vmovaps   %zmm13, %zmm29                                #2004.33 c5
..LN8491:
	.loc    1  2021  is_stmt 1
        vmovaps   %zmm11, %zmm0                                 #2021.19 c7
..LN8492:
	.loc    1  2031  is_stmt 1
        vpaddq    %zmm16, %zmm15, %zmm15                        #2031.16 c7
..LN8493:
	.loc    1  1990  is_stmt 1
        vgatherdpd z(,%ymm1,8), %zmm20{%k1}                     #1990.27 c7
..LN8494:
	.loc    1  1991  is_stmt 1
        vgatherdpd 8+z(,%ymm1,8), %zmm21{%k2}                   #1991.27 c7
..LN8495:
	.loc    1  1994  is_stmt 1
        vsubpd    %zmm9, %zmm20, %zmm3                          #1994.24 c13 stall 2
..LN8496:
	.loc    1  1995  is_stmt 1
        vsubpd    %zmm8, %zmm21, %zmm2                          #1995.24 c13
..LN8497:
	.loc    1  1992  is_stmt 1
        vgatherdpd 16+z(,%ymm1,8), %zmm22{%k3}                  #1992.27 c13
..LN8498:
	.loc    1  1997  is_stmt 1
        vmulpd    %zmm3, %zmm3, %zmm28                          #1997.24 c19 stall 2
..LN8499:
	.loc    1  1996  is_stmt 1
        vsubpd    %zmm6, %zmm22, %zmm31                         #1996.24 c19
..LN8500:
	.loc    1  1997  is_stmt 1
        vfmadd231pd %zmm2, %zmm2, %zmm28                        #1997.24 c25 stall 2
..LN8501:
	.loc    1  2019  is_stmt 1
        vmovaps   %zmm11, %zmm20                                #2019.19 c25
..LN8502:
	.loc    1  2020  is_stmt 1
        vmovaps   %zmm11, %zmm21                                #2020.19 c27
..LN8503:
	.loc    1  1997  is_stmt 1
        vfmadd231pd %zmm31, %zmm31, %zmm28                      #1997.24 c31 stall 1
..LN8504:
	.loc    1  2003  is_stmt 1
        vmulpd    %zmm28, %zmm28, %zmm23                        #2003.47 c37 stall 2
..LN8505:
	.loc    1  2011  is_stmt 1
        vcmppd    $2, %zmm7, %zmm28, %k1{%k4}                   #2011.27 c37
..LN8506:
	.loc    1  2019  is_stmt 1
        kmovw     %k1, %k5                                      #2019.19 c39
..LN8507:
	.loc    1  2020  is_stmt 1
        kmovw     %k1, %k6                                      #2020.19 c39
..LN8508:
	.loc    1  2021  is_stmt 1
        kmovw     %k1, %k7                                      #2021.19 c41
..LN8509:
	.loc    1  2019  is_stmt 1
        vgatherdpd 32+z(,%ymm1,8), %zmm20{%k5}                  #2019.19 c41
..LN8510:
	.loc    1  2027  is_stmt 1
        kmovw     %k1, %k5                                      #2027.7 c41
..LN8511:
	.loc    1  2020  is_stmt 1
        vgatherdpd 40+z(,%ymm1,8), %zmm21{%k6}                  #2020.19 c41
..LN8512:
	.loc    1  2003  is_stmt 1
        vmulpd    %zmm23, %zmm28, %zmm24                        #2003.33 c43
..LN8513:
	.loc    1  2028  is_stmt 1
        kmovw     %k1, %k6                                      #2028.7 c43
..LN8514:
	.loc    1  2021  is_stmt 1
        vgatherdpd 48+z(,%ymm1,8), %zmm0{%k7}                   #2021.19 c47 stall 1
..LN8515:
	.loc    1  2005  is_stmt 1
        vmulpd    %zmm24, %zmm24, %zmm25                        #2005.47 c49
..LN8516:
	.loc    1  2004  is_stmt 1
        vfmsub231pd %zmm24, %zmm14, %zmm29                      #2004.33 c49
..LN8517:
	.loc    1  2005  is_stmt 1
        vmulpd    %zmm25, %zmm28, %zmm26                        #2005.33 c55 stall 2
..LN8518:
	.loc    1  2006  is_stmt 1
        vrcp28pd  %zmm26, %zmm27                                #2006.33 c61 stall 2
..LN8519:
	.loc    1  2007  is_stmt 1
        vfnmadd213pd %zmm12, %zmm27, %zmm26                     #2007.33 c69 stall 3
..LN8520:
	.loc    1  2008  is_stmt 1
        vmulpd    %zmm27, %zmm26, %zmm30                        #2008.33 c75 stall 2
..LN8521:
	.loc    1  2013  is_stmt 1
        vmulpd    %zmm30, %zmm29, %zmm22{%k1}{z}                #2013.13 c81 stall 2
..LN8522:
	.loc    1  2015  is_stmt 1
        vfmadd231pd %zmm22, %zmm3, %zmm18                       #2015.14 c87 stall 2
..LN8523:
	.loc    1  2016  is_stmt 1
        vfmadd231pd %zmm22, %zmm2, %zmm19                       #2016.14 c87
..LN8524:
	.loc    1  2017  is_stmt 1
        vfmadd231pd %zmm22, %zmm31, %zmm10                      #2017.14 c93 stall 2
..LN8525:
	.loc    1  2023  is_stmt 1
        vfnmadd231pd %zmm22, %zmm3, %zmm20                      #2023.14 c93
..LN8526:
	.loc    1  2027  is_stmt 1
        vscatterdpd %zmm20, 32+z(,%ymm1,8){%k5}                 #2027.7 c99 stall 2
..LN8527:
	.loc    1  2024  is_stmt 1
        vfnmadd231pd %zmm22, %zmm2, %zmm21                      #2024.14 c99
..LN8528:
	.loc    1  2028  is_stmt 1
        vscatterdpd %zmm21, 40+z(,%ymm1,8){%k6}                 #2028.7 c105 stall 2
..LN8529:
	.loc    1  2025  is_stmt 1
        vfnmadd213pd %zmm0, %zmm31, %zmm22                      #2025.14 c105
..LN8530:
	.loc    1  2029  is_stmt 1
        vscatterdpd %zmm22, 48+z(,%ymm1,8){%k1}                 #2029.7 c111 stall 2
..LN8531:
	.loc    1  1986  is_stmt 1
        cmpl      %r10d, %r11d                                  #1986.25 c111
..LN8532:
        jl        ..B34.5       # Prob 82%                      #1986.25 c113
..LN8533:
                                # LOE rax rdx rcx rbx rbp rsi rdi r8 r9 r12 r13 r14 r15 r10d r11d zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19
..B34.7:                        # Preds ..B34.5 ..B34.3
                                # Execution count [3.57e+05]
..LN8534:
	.loc    1  2035  is_stmt 1
        vextractf64x4 $1, %zmm19, %ymm6                         #2035.17 c1
..LN8535:
        vmovaps   %zmm19, %zmm8                                 #2035.17 c1
..LN8536:
	.loc    1  1971  is_stmt 1
        addq      $1, %rdi                                      #1971.27 c1
..LN8537:
	.loc    1  2035  is_stmt 1
        vaddpd    %ymm8, %ymm6, %ymm9                           #2035.17 c3
..LN8538:
	.loc    1  2034  is_stmt 1
        vextractf64x4 $1, %zmm18, %ymm0                         #2034.17 c3
..LN8539:
        vmovaps   %zmm18, %zmm1                                 #2034.17 c5
..LN8540:
        vaddpd    %ymm1, %ymm0, %ymm2                           #2034.17 c7
..LN8541:
	.loc    1  2035  is_stmt 1
        valignq   $3, %zmm9, %zmm9, %zmm19                      #2035.17 c11
..LN8542:
        valignq   $2, %zmm9, %zmm9, %zmm24                      #2035.17 c13
..LN8543:
        valignq   $1, %zmm9, %zmm9, %zmm25                      #2035.17 c13
..LN8544:
	.loc    1  2036  is_stmt 1
        vextractf64x4 $1, %zmm10, %ymm1                         #2036.17 c19
..LN8545:
	.loc    1  2035  is_stmt 1
        vaddsd    %xmm9, %xmm19, %xmm0                          #2035.17 c21
..LN8546:
        vaddsd    %xmm25, %xmm24, %xmm15                        #2035.17 c21
..LN8547:
        vaddsd    %xmm0, %xmm15, %xmm26                         #2035.17 c27 stall 2
..LN8548:
	.loc    1  2036  is_stmt 1
        vaddpd    %ymm10, %ymm1, %ymm0                          #2036.17 c27
..LN8549:
	.loc    1  2035  is_stmt 1
        vaddsd    40+z(%rsi), %xmm26, %xmm27                    #2035.5 c33
..LN8550:
        vmovsd    %xmm27, 40+z(%rsi)                            #2035.5 c39 stall 2
..LN8551:
	.loc    1  2034  is_stmt 1
        valignq   $3, %zmm2, %zmm2, %zmm18                      #2034.17 c39
..LN8552:
        valignq   $2, %zmm2, %zmm2, %zmm20                      #2034.17 c39
..LN8553:
        valignq   $1, %zmm2, %zmm2, %zmm21                      #2034.17 c41
..LN8554:
	.loc    1  2036  is_stmt 1
        valignq   $3, %zmm0, %zmm0, %zmm28                      #2036.17 c41
..LN8555:
        valignq   $2, %zmm0, %zmm0, %zmm29                      #2036.17 c43
..LN8556:
        valignq   $1, %zmm0, %zmm0, %zmm30                      #2036.17 c43
..LN8557:
	.loc    1  2034  is_stmt 1
        vaddsd    %xmm2, %xmm18, %xmm5                          #2034.17 c53
..LN8558:
        vaddsd    %xmm21, %xmm20, %xmm3                         #2034.17 c53
..LN8559:
	.loc    1  2036  is_stmt 1
        vaddsd    %xmm0, %xmm28, %xmm2                          #2036.17 c59 stall 2
..LN8560:
        vaddsd    %xmm30, %xmm29, %xmm1                         #2036.17 c59
..LN8561:
	.loc    1  2034  is_stmt 1
        vaddsd    %xmm5, %xmm3, %xmm22                          #2034.17 c65 stall 2
..LN8562:
	.loc    1  2036  is_stmt 1
        vaddsd    %xmm2, %xmm1, %xmm31                          #2036.17 c65
..LN8563:
	.loc    1  2034  is_stmt 1
        vaddsd    32+z(%rsi), %xmm22, %xmm23                    #2034.5 c71 stall 2
..LN8564:
        vmovsd    %xmm23, 32+z(%rsi)                            #2034.5 c77 stall 2
..LN8565:
	.loc    1  2036  is_stmt 1
        vaddsd    48+z(%rsi), %xmm31, %xmm0                     #2036.5 c77
..LN8566:
        vmovsd    %xmm0, 48+z(%rsi)                             #2036.5 c83 stall 2
..LN8567:
	.loc    1  1971  is_stmt 1
        addq      $64, %rsi                                     #1971.27 c83
..LN8568:
        cmpq      %r8, %rdi                                     #1971.23 c83
..LN8569:
        jl        ..B34.3       # Prob 99%                      #1971.23 c85
..LN8570:
                                # LOE rdx rcx rbx rbp rsi rdi r8 r9 r12 r13 r14 r15 zmm4 zmm7 zmm11 zmm12 zmm13 zmm14 zmm16 zmm17
..B34.9:                        # Preds ..B34.7 ..B34.1
                                # Execution count [1.00e+00]
..LN8571:
	.loc    1  2038  epilogue_begin  is_stmt 1
        ret                                                     #2038.1 c3
        .align    16,0x90
..LN8572:
                                # LOE
..LN8573:
	.cfi_endproc
# mark_end;
	.type	force_intrin_v6_noswp(),@function
	.size	force_intrin_v6_noswp(),.-force_intrin_v6_noswp()
..LN_Z21force_intrin_v6_noswpv.8574:
.LN_Z21force_intrin_v6_noswpv:
	.data
# -- End  force_intrin_v6_noswp()
