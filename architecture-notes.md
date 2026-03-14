- changed the embeding model to nomic to snowflake

### can do better 
- we can do better by doing chunking and embedding in parallel 
- run third party models instead of OSS models 

architecture notes:
- end point for injestion is fast_injest.py is being called by run_e2e_test.py
- fast_injesty.py is calling go injestion 
- 