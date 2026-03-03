package sk.ainet.apps.kllama.chat.java

import sk.ainet.apps.kllama.chat.*
import sk.ainet.apps.kllama.java.KLlamaSession
import java.util.function.Consumer

/**
 * Java-friendly wrapper around the agent loop for tool-calling conversations.
 *
 * Example usage from Java:
 * ```java
 * JavaAgentLoop agent = JavaAgentLoop.builder()
 *     .session(session)
 *     .tool(myCalculatorTool)
 *     .tool(mySearchTool)
 *     .systemPrompt("You are a helpful assistant with tools.")
 *     .build();
 *
 * String response = agent.chat("What is 42 * 17?");
 * ```
 */
public class JavaAgentLoop private constructor(
    private val session: KLlamaSession,
    private val tools: List<JavaTool>,
    private val systemPrompt: String,
    private val config: AgentConfig,
    private val templateName: String
) {
    private val messages = mutableListOf<ChatMessage>()
    private val toolRegistry = ToolRegistry()

    init {
        tools.forEach { toolRegistry.register(JavaToolAdapter(it)) }
        messages.add(ChatMessage(role = ChatRole.SYSTEM, content = systemPrompt))
    }

    /**
     * Send a user message and get the agent's response (may include tool calls).
     *
     * @param userMessage The user's message.
     * @return The final assistant response.
     */
    public fun chat(userMessage: String): String {
        messages.add(ChatMessage(role = ChatRole.USER, content = userMessage))

        val template: ChatTemplate = when (templateName.lowercase()) {
            "chatml" -> ChatMLTemplate()
            else -> Llama3ChatTemplate()
        }

        val agentLoop = AgentLoop(
            runtime = session.runtime,
            template = template,
            toolRegistry = toolRegistry,
            eosTokenId = session.tokenizer.let {
                if (it is sk.ainet.apps.kllama.GGUFTokenizer) it.eosId else 2
            },
            config = config,
            decode = { session.tokenizer.decode(it) }
        )

        return agentLoop.runWithEncoder(
            messages = messages,
            encode = { session.tokenizer.encode(it) }
        )
    }

    /**
     * Send a user message with streaming and get the final response.
     *
     * @param userMessage The user's message.
     * @param tokenConsumer Called for each generated token fragment.
     * @return The final assistant response.
     */
    public fun chat(userMessage: String, tokenConsumer: Consumer<String>): String {
        messages.add(ChatMessage(role = ChatRole.USER, content = userMessage))

        val template: ChatTemplate = when (templateName.lowercase()) {
            "chatml" -> ChatMLTemplate()
            else -> Llama3ChatTemplate()
        }

        val agentLoop = AgentLoop(
            runtime = session.runtime,
            template = template,
            toolRegistry = toolRegistry,
            eosTokenId = session.tokenizer.let {
                if (it is sk.ainet.apps.kllama.GGUFTokenizer) it.eosId else 2
            },
            config = config,
            decode = { session.tokenizer.decode(it) }
        )

        val listener = object : AgentListener {
            override fun onToken(token: String) {
                tokenConsumer.accept(token)
            }
        }

        return agentLoop.runWithEncoder(
            messages = messages,
            encode = { session.tokenizer.encode(it) },
            listener = listener
        )
    }

    /** Reset conversation history (keeps system prompt). */
    public fun reset() {
        messages.clear()
        messages.add(ChatMessage(role = ChatRole.SYSTEM, content = systemPrompt))
    }

    public companion object {
        @JvmStatic
        public fun builder(): Builder = Builder()
    }

    public class Builder {
        private var session: KLlamaSession? = null
        private val tools = mutableListOf<JavaTool>()
        private var systemPrompt: String = "You are a helpful assistant."
        private var config: AgentConfig = AgentConfig()
        private var templateName: String = "llama3"

        public fun session(session: KLlamaSession): Builder {
            this.session = session
            return this
        }

        public fun tool(tool: JavaTool): Builder {
            tools.add(tool)
            return this
        }

        public fun systemPrompt(systemPrompt: String): Builder {
            this.systemPrompt = systemPrompt
            return this
        }

        public fun config(config: AgentConfig): Builder {
            this.config = config
            return this
        }

        public fun template(templateName: String): Builder {
            this.templateName = templateName
            return this
        }

        public fun build(): JavaAgentLoop {
            return JavaAgentLoop(
                session = requireNotNull(session) { "session must be set" },
                tools = tools.toList(),
                systemPrompt = systemPrompt,
                config = config,
                templateName = templateName
            )
        }
    }
}
