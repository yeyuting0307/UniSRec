## UniSRec: Unified Sequential Recommendation

This repository provides the source code for **UniSRec**, a unified sequential recommendation framework that bridges the gap between text-aware and conventional sequential recommendation.

**Key Features:**

* **Unified architecture:** Supports both text-aware and conventional sequential recommendation models within a single framework.
* **Flexibility:** Easily switch between different pre-trained language models and sequential recommendation models.

**Models:**

* **UniSRec:**  A unified sequential recommendation model that leverages textual item descriptions and user-item interaction sequences.

**Code Structure:**

* **models:** Contains the implementation of UniSRec and other baseline models.
  * **unisrec.py:** Implements the core UniSRec model.
  * **sasrec.py:** Implements the SASRec model, used as a component within UniSRec.

**Reference:**
  * arxiv: https://arxiv.org/abs/2206.05941