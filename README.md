# EduLLM: Applying Large Language Models in the Education Industry

![icon](.\asset\Edullm_icon.png)

## Progress

- [x] 🎯📢04/03 - 04/06: Implemented the new RAG method.  
- [x] 🎯📢04/06: Completed the algorithm development.  
- [x] 🎯📢04/10: Finalized both frontend and backend implementation.  
- [x] 🎯📢04/18: Completed all project evaluations.  

**🎉 Project Status: Successfully Completed! 🎉**

---

## Algorithm Design

### Building the Database:
- Pass all questions from the dataset \( Q = \{q_1, q_2, \ldots, q_n\} \) into the database \( MEM \).
- For each question \( q_i \), extract its corresponding **knowledge points** \( K_i = \{[k_i^1, k_i^2, \ldots, k_i^p]\} \), where \( p \) is a predefined hyperparameter.
- For each knowledge point \( k \in K_i \), convert it into a vector \( v \in V_i \) using a text embedding function \( \text{Embedder} \):  
  \( V_i = \text{Embedder}(K_i) \).  
  - If \( |V_i| \neq p \), pad with \( \vec{0} \) until \( |V_i| = p \).
- For each question \( q_i \), create a memory entry \( \text{memory}_q\{q_i, V_i, K_i\} \) and store it in \( MEM \).
- The final database becomes \( MEM = \{\text{memory}_1, \ldots, \text{memory}_n\} \).

---

### Retrieving from the Database:
- For an input question \( q_{\text{example}} \), extract its knowledge points \( K_{\text{example}} \).
- Convert \( K_{\text{example}} \) into vectors \( V_{\text{example}} \) using \( \text{Embedder} \). Pad with \( \vec{0} \) if \( |V_{\text{example}}| \neq p \).
- For each vector \( v_{\text{example}}^i \in V_{\text{example}} \), compute the dot-product similarity with all vectors in \( MEM \) (i.e., \( [V_1, V_2, \ldots, V_n] \)).
- Sum all pairwise similarities for each \( q_i \) in \( MEM \):  
  \( \text{sim} = \text{pairwise}(V_i, V_{\text{example}}) \), then take the average: \( \text{sum}(\text{sim}) / p^2 \).
- Return the \( \text{topK} \) most similar questions and their associated knowledge points.

---

### Generating Questions:
- Construct a `system_prompt`.
- Add the \( \text{topK} \) similar questions and their knowledge points to `system_prompt`.
- Include only the knowledge points \( K_{\text{example}} \) of \( q_{\text{example}} \) in `system_prompt` to form the `final_prompt`.
- Generate the output question: \( q_{\text{output}} = \text{LLM}(\text{final\_ prompt}) \).

---

### Environment Setup:
```bash
conda create -n edullm python=3.12
conda activate edullm
pip install gdm-concordia
pip install -U sentence-transformers
pip install reportlab
```

---

### Chinese version:
- [中文版](.\asset\chinese_readme\readme_cn.md)

<!-- ## How to run:

1. Create Environment

### if you want to generate past paper and evaluate
#### generate pastpaper
- `cd EduLLM\agent`
- go to `EduLLM\agent\main_function_for_evaluation.py`
  - setup the data filename at line:318 - 328
  - `file_names` and `data_path`
  - setup the method to be used at line:339 - 352
    - e.g. `_generate_question_from_pastpaper_with_knowledge_point_and_bloom`
- generated past paper can be found at `agent/output` folder
#### evaluation
- copy the generated past paper to `agent/evaluation` folder
- `cd agent/evaluation`
- open the `llm_as_a_judge.py`
- setup `original_path` at line210-214
- setup `get_original_question` function at line 184-185
- run the  `llm_as_a_judge.py` at in folder `evaluation`
- the result will be output in `result_0417` or you can setup target path by yourself, the filename of the result is `evaluation.json`
- then we can open the `analysis.ipynb` and setup the path to `evaluation.json` 
- then you can run all to get the final analysis, including the winning rate. -->
  
## How to Run:

1. Create Environment

### If You Want to Generate Past Papers and Evaluate

**Generate Past Papers**
- `cd EduLLM/agent`
- Go to `EduLLM/agent/main_function_for_evaluation.py`.
  - Set up the data filename at lines 318–328:
    - `file_names` and `data_path`
  - Set up the method to be used at lines 339–352:
    - For example: `_generate_question_from_pastpaper_with_knowledge_point_and_bloom`
- The generated past papers can be found in the `agent/output` folder.

**Evaluation**
- Copy the generated past papers to the `agent/evaluation` folder.
- `cd agent/evaluation`
- Open `llm_as_a_judge.py`.
- Set up `original_path` at lines 210–214.
- Set up the `get_original_question` function at lines 184–185.
- Run `llm_as_a_judge.py` in the `evaluation` folder.
- The result will be output in `result_0417`, or you can set a custom target path. The filename of the result is `evaluation.json`.
- Then, open `analysis.ipynb` and set the path to `evaluation.json`.
- Run all cells to get the final analysis, including the winning rate.

### If you want to use the UI
- `cd web.py`
- run `web.py`
- then click the localhost link