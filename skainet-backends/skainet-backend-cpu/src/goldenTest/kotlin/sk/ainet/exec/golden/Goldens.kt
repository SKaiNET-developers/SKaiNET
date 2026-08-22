package sk.ainet.exec.golden

/**
 * Recorded golden digests (see [GoldenSupport.digest]). One entry per (test, encoding). Recorded on
 * JVM (HotSpot, strict binary32) and verified bit-identical on Kotlin/Native linuxX64.
 *
 * Re-baselining is a deliberate act: change a value only in the PR that intentionally changes the
 * numeric behaviour of a decoder or scalar kernel, and say so in the PR.
 */
internal object Goldens {
    val expected: Map<String, String> = mapOf(
        "decode/Q4_0" to "n=384 fnv=d6bc5d718cd98a58 head=bdb0dc00,3d979800,00000000,3d4a2000",
        "decode/Q4_K" to "n=3072 fnv=fa096e860a45191e head=4049e820,408a0210,409c8910,40c19710",
        "decode/Q5_0" to "n=384 fnv=6a72d36758b74185 head=3dc3e000,bdc3e000,bef4d800,be74d800",
        "decode/Q5_1" to "n=384 fnv=b33abdb77cf4b87f head=be80e800,bd89d000,3bd50000,3ead5800",
        "decode/Q5_K" to "n=3072 fnv=3381529c397a361d head=4125da40,416af240,41980520,41980520",
        "decode/Q6_K" to "n=3072 fnv=3c3d7fa00c48faa9 head=c2606550,40ef5b00,41fe50b0,42606550",
        "decode/Q8_0" to "n=384 fnv=334c03e32df05c28 head=3f6a1700,bf6ede00,3ee55000,bf786c00",
        "decode/TERNARY_tq2_0" to "n=256 fnv=48409a8f0eeaba6e head=bd000000,bd000000,bd000000,bd000000",
        "decode/TERNARY_values" to "n=384 fnv=e455e2a9df6acf48 head=00000000,bf500000,bf500000,3f500000",
        "scalar-matmul/Q4_0" to "n=12 fnv=0aaee3d77e3619f6 head=3f15be82,3f0f4b4e,3e49fb8e,be220b31",
        "scalar-matmul/Q4_K" to "n=12 fnv=1c27f69421ebce94 head=c29e9232,c3116957,c3a2cb4f,c2da54e5",
        "scalar-matmul/Q5_0" to "n=12 fnv=9cf4c45d4329bee6 head=3ffafe21,4073853c,c0a97fbe,3e9fe3a0",
        "scalar-matmul/Q5_1" to "n=12 fnv=3a57aa4feea433fd head=3fe5eab0,bfe35b0c,40821cc5,40ca2f5a",
        "scalar-matmul/Q5_K" to "n=12 fnv=69764ad908eba4c2 head=4317c800,c3350201,4380baa3,4286212d",
        "scalar-matmul/Q6_K" to "n=12 fnv=b534bb781b931bb2 head=c41704aa,425d6ccb,4423cdf0,43bffcfd",
        "scalar-matmul/Q8_0" to "n=12 fnv=a1ba10e77b91e0ab head=bf462edd,3f92c634,c151b4cb,c0e14735",
        "turboquant/polar3/codes" to "n=48 fnv=46f69505e290de9f head=5db9a5a5d84adcb4",
        "turboquant/polar4-qjl1/codes" to "n=64 fnv=4e1c32b146ebeda5 head=6a7b76918a83ba43",
        "turboquant/polar4/codes" to "n=64 fnv=4e1c32b146ebeda5 head=6a7b76918a83ba43",
        "packed/TERNARY_values" to "n=96 fnv=f659cade8a7a5701 head=8192058611602495",
        "turboquant/polar3/scales" to "n=4 fnv=91fae3204b7f58d3 head=3fa460a9,3f9ffba5,400b2d5d,3fdc5844",
        "turboquant/polar4-qjl1/scales" to "n=4 fnv=296a63c84523f465 head=3f0ce523,3f0920d7,3f6e96e8,3f3cddf1",
        "turboquant/polar4/scales" to "n=4 fnv=296a63c84523f465 head=3f0ce523,3f0920d7,3f6e96e8,3f3cddf1",
        "turboquant/polar3/decoded" to "n=128 fnv=92a4aba1adab2736 head=3ef321a3,be3f44b9,bf927550,4054d201",
        "turboquant/polar4-qjl1/decoded" to "n=128 fnv=c709e3b423fd49ff head=3ef331d0,beddd096,bf501ad5,404d9967",
        "turboquant/polar4/decoded" to "n=128 fnv=aad39f2f7c413875 head=3ed15f9f,bebbfe6a,bf71ed00,403cb050",
    )
}
