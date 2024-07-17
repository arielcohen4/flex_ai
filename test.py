from flex_ai import FlexAI, DatasetType

client = FlexAI(api_key="ed8ef09e-3ca0-4080-bd93-761fa5428d65")

# client.create_dataset("API Dataset New","instruction/train.jsonl", "instruction/eval.jsonl", DatasetType.INSTRUCTION)

client.create_finetune(name="My Task New", 
                       dataset_id="62bf5ae8-bbba-438b-9564-36bc902ab393",
                       model="TinyLlama/TinyLlama_v1.1",
                       n_epochs=5, 
                       train_with_lora=True,
                       lora_config={"lora_r": 16, "lora_alpha": 8, "lora_dropout": 0.1}, 
                       n_checkpoints_and_evaluations_per_epoch=1)

# client.create_finetune(name="My Task New", 
#                        dataset_id="52c68b52-12f1-4c51-97ac-4434c85357e2",
#                        model="meta-llama/Meta-Llama-3-8B-Instruct",
#                        n_epochs=5, batch_size=4, learning_rate=0.0001, n_checkpoints_and_evaluations_per_epoch=1, 
#                        save_only_best_checkpoint=True, train_with_lora=True, 
#                        lora_config={"lora_r": 64, "lora_alpha": 8, "lora_dropout": 0.1}, 
#                        early_stopping_config={"patience": 1, "threshold": 0.1})