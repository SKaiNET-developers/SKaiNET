---
name: "DARC Feature Proposal"
about: "Propose a new feature, improvement, or component using the DARC methodology"
title: "[Feature]: "
labels: enhancement
assignees: ""
---

# 🧠 D: DOCUMENT — Problem & Opportunity

**What is the problem, limitation, or opportunity? Why does this matter for SKaiNET?**

Explain the context in clear terms:
- What functionality is missing or insufficient?
- Why is this important now?
- Who benefits (users, developers, researchers)?
- What is the high-level goal of this proposal?

**Summary (1–2 sentences):**  
> _Provide a concise overview of what this issue aims to address._

---

# 🔍 A: ASSESS — Feasibility & Impact

Provide an evaluation of the proposal. Address the following:

### ✔️ Feasibility
- Is the feature straightforward to implement?
- Are there architectural constraints?

### ✔️ Expected Impact
- How does this improve SKaiNET?
- Does it unlock new capabilities?

### ✔️ Risks / Constraints
- Technical challenges?
- Numerical stability concerns?
- API consistency?

### ✔️ Dependencies
- Does this rely on an existing SKaiNET module?
- Does it require third-party libraries or standards?

---

# 📚 R: RESEARCH — What Must Be Understood First?

Document research tasks or open questions that must be answered before implementation:

### Research Tasks
- [ ] Review existing frameworks (PyTorch, JAX, TensorFlow, etc.)
- [ ] Identify relevant algorithms or formulas
- [ ] Compare design patterns for modularity
- [ ] Investigate numerically stable or optimized variants
- [ ] Check for relevant academic papers or benchmarks

### Open Questions
> _List unknowns that contributors should discuss or resolve._

---

# 🛠️ C: CODE — Implementation Plan

Break down actionable steps required to deliver this feature:

### Development Tasks
- [ ] Design module interface
- [ ] Implement core functionality
- [ ] Write unit tests (correctness + edge cases)
- [ ] Add configuration integration
- [ ] Update training loop or pipeline components (if applicable)
- [ ] Add documentation & examples
- [ ] Create or update benchmarks (optional)
- [ ] Submit PR referencing this issue

### Acceptance Criteria
- [ ] Feature works as intended and passes all tests
- [ ] Fully documented with clear examples
- [ ] Follows SKaiNET coding style & modular design
- [ ] Reviewed and approved by maintainers
- [ ] No regressions introduced elsewhere

---

# 💬 Additional Notes

> _Add diagrams, pseudo-code, references, or implementation notes that may help future contributors._
