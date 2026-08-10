# ServiceLoader discovery of the JNI kernel provider (#920).
#
# R8 full mode strips classes that are only referenced from
# META-INF/services files. Without these rules a RELEASE build silently
# loses the provider and inference falls back to the 100x-slower scalar
# path with no error — the exact failure mode these rules exist to prevent.
-keep class sk.ainet.exec.kernel.jni.JniKernelProviderFactory { *; }
-keep class sk.ainet.exec.kernel.jni.JniKernels { *; }

# Native method registration: JNI resolves Java_sk_ainet_exec_kernel_jni_*
# symbols against these exact names — neither class nor method names may be
# renamed or removed.
-keepclasseswithmembers class sk.ainet.exec.kernel.jni.JniKernels {
    native <methods>;
}
