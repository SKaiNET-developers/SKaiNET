package sk.ainet.apps.knanogpt

interface Tokenizer<T> {

    fun encode(text: String): List<T>

    fun decode(tokens: List<T>): String
}

fun main() {
    // Step 1. Import ..
/*
    // Step 2. Initialisierung-Block
    //how many independent sequences will we process in parallel?
    val batchSize = 4
    // what is the maximum context length for predictions?
    val blockSize = 8


    // Step 3. 4.  Load & tokenize data
    val dataProvider = ResourcesDataProvider("input.txt")
    val data = dataProvider.load()
    assert(
        data.shape.dimensions.size == 1
    ) { "`shape.dimensions.size` must be ${data.shape.dimensions.size}" }

    // Step 5. Train & Test Spilt
    val splitter = TrainTestSplitter(data)
    val (trainData, testData) = splitter.split(0.9f)
    println(trainData.shape.volume)
    println(testData.shape.volume)
    assert(
        trainData.shape.dimensions.size == 1
    ) { "`shape.dimensions.size` must be ${trainData.shape.dimensions.size}" }
    // Step 6. get batch
    val batch = BatchProvider(trainData, testData, blockSize, batchSize)
    val (xb, yb) = batch.getRandomBatch(true)

    print("inputs:")
    print(xb.shape)
    print(xb)
    print("targets:")
    print(yb.shape)
    print(yb)

    repeat(batchSize) { b ->
        repeat(blockSize) { t ->
            val context = xb[b..b, 0..t]
            val target = yb[b, t]
            println("when input is $context the target: $target")
        }
    }

 */
}