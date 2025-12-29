package sk.ainet.compile.c

/**
 * Arduino C code generation package for SKaiNET neural networks.
 * 
 * This package provides functionality to convert SKaiNET ComputeGraph instances
 * into C99-compatible code suitable for Arduino microcontrollers.
 */

/**
 * Platform-specific directory creation.
 */
public expect fun platformCreateDirectory(path: String)

/**
 * Platform-specific file writing.
 */
public expect fun platformWriteFile(path: String, content: String)