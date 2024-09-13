### **DejaVu Chatbot Model (Gemma-2-2B-IT)**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14098797%2F7669fdc2a2d64e8d511947ea88c4b238%2F37110581-A303-4A6D-B47E3F1844F0707D_source.webp?generation=1727999417326204&alt=media)

**Model Overview**:
The DejaVu chatbot is a conversational AI designed to assist users in recalling forgotten search queries. It leverages the **Gemma-2-2B-IT** language model, a 2-billion parameter causal language model pre-trained on diverse datasets, and fine-tuned specifically for contextual memory recall. This chatbot is built to help users remember what they intended to search for by asking about recent activities, interactions, or ongoing tasks and generating intelligent search suggestions based on these clues.

**Key Features**:
1. **Contextual Memory Recall**: 
   The model is trained to help users remember forgotten search queries by utilizing contextual information such as recent activities, interactions, tasks, and environment. Users often experience moments of memory lapse where they know they had something to search for but can't recall the exact query. DejaVu bridges that gap by analyzing user-provided context and generating relevant search suggestions.

2. **Personalized Conversations**: 
   By understanding the user’s recent actions and conversations, the chatbot dynamically adapts its responses to each user’s unique context. For example, it can recall tasks you were involved in, such as "watching movie reviews" or "talking with a friend," and suggest potential searches based on those activities.

3. **Real-Time Suggestion Generation**: 
   Using the **Gemma-2-2B-IT** model, DejaVu generates search suggestions in real-time with high accuracy. The chatbot generates plausible search queries in under 150 tokens, ensuring that users are quickly assisted without overwhelming them with irrelevant information.

4. **Efficient Resource Utilization**:
   The DejaVu chatbot takes advantage of 4-bit quantization and low-memory usage techniques, ensuring the model can be efficiently deployed on resource-constrained systems. It can leverage GPU acceleration where available, providing faster response times.

**Training and Fine-Tuning**:
The model was fine-tuned on a custom dataset, which included user interactions and past searches, to specialize in search recall. The training was done using a LoRA-based method for efficient parameter fine-tuning, allowing the model to generalize well while being lightweight. Key hyperparameters such as learning rate, batch size, and warmup steps were fine-tuned for optimal performance on this task.

**Use Case Example**:
- **User Input**: "I was just talking with a friend about a movie, but I can’t remember what I wanted to search for."
- **Generated Output**: "Were you looking for information on the new Christopher Nolan movie? Maybe 'Oppenheimer'?"

**Applications**:
- **Search Engines**: DejaVu can be integrated into search platforms to improve user experience by helping users retrieve forgotten search queries.
- **Personal Assistants**: It can be embedded in virtual assistants, helping users remember their to-dos or planned searches.
- **Productivity Tools**: DejaVu can assist in recalling forgotten tasks, documents, or internet searches in professional environments.

**Model Technical Specifications**:
- **Model**: Gemma-2-2B-IT (2 billion parameters)
- **Architecture**: Causal Language Model (GPT-style)
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for efficient usage
- **Maximum Sequence Length**: 1024 tokens
- **Pretrained On**: A diverse corpus including internet text, ensuring general understanding of various topics and languages.

For more details on the fine-tuning process and implementation of this model, you can check out the notebook: https://www.kaggle.com/code/kyungwanwoo/dejavu
