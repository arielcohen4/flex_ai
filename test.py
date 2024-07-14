from flex_ai import FlexAI, DatasetType

aaa = FlexAI(api_key="ed8ef09e-3ca0-4080-bd93-761fa5428d65")

aaa.create_dataset("My Dataset New","instruction/train.jsonl", "instruction/eval.jsonl", DatasetType.INSTRUCTION)