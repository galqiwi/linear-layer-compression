import torch

from transformers import AutoModelForCausalLM, HqqConfig

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=3, group_size=1024)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B", 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)

def get_zero_shots(model, task_list = ('arc_easy',), num_fewshots=1):
    import lm_eval

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
    )

    tasks = lm_eval.tasks.get_task_dict(task_list)
    if num_fewshots != 1:
        # TODO: make fewshots properly
        for task_name in tasks:
            task = tasks[task_name]
            if isinstance(task, tuple):
                task = task[1]
            if task is None:
                continue
            task.config.num_fewshot = num_fewshots

    results = lm_eval.evaluator.evaluate(
        lm=lm_eval_model,
        task_dict=tasks,
    )

    result_dict = {task_name: task_result['acc,none'] for task_name, task_result in results['results'].items()}
    result_err_dict = {f'{task_name}_err': task_result['acc_stderr,none'] for task_name, task_result in
                       results['results'].items()}
    result_dict = dict(list(result_dict.items()) + list(result_err_dict.items()))

    if num_fewshots != 1:
        result_dict = {f'{task_name}@{num_fewshots}': acc for task_name, acc in result_dict.items()}

    return result_dict

results = get_zero_shots(
    model,
    task_list=("mmlu",),
    num_fewshots=5,
)
# results = get_zero_shots(model, task_list = ('winogrande','piqa','arc_easy','arc_challenge'), num_fewshots=1)


print(results)
