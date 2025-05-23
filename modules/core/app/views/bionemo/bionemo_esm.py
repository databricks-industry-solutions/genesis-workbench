import streamlit as st

st.title(":material/genetics: BioNeMo - ESMFold2")

settings_tab, finetune_tab, inference_tab = st.tabs(["Settings","Fine Tune", "Inference"])

with finetune_tab:
    with st.form("finetune_esm_form", enter_to_submit=False ):
        c1,c2,c3 = st.columns([1,1,1], vertical_alignment="bottom")
        with c1:
            esm_variant = st.selectbox("ESM Variant",["650M","3B"])
            train_data = st.text_input("Train Data Path:")
            evaluation_data = st.text_input("Evaluation Data Path:")
            should_use_lora =  st.toggle("Use LoRA?") 
            finetune_label = st.text_input("Finetuning Label:")
            start_finetune_button_clicked = st.form_submit_button("Finetune")


        

train_data_volume_location = dbutils.widgets.get("train_data_location")
validation_data_volume_location = dbutils.widgets.get("validation_data_location")
should_use_lora = True if dbutils.widgets.get("should_use_lora")=="true" else False
finetune_label = dbutils.widgets.get("finetune_label")
task_type = dbutils.widgets.get("task-type")
mlp_ft_dropout = float(dbutils.widgets.get("mlp-ft-dropout"))
mlp_hidden_size = int(dbutils.widgets.get("mlp-hidden-size"))
mlp_target_size = int(dbutils.widgets.get("mlp-target-size"))
experiment_name = dbutils.widgets.get("experiment-name")
num_steps = int(dbutils.widgets.get("num-steps"))
lr = float(dbutils.widgets.get("lr"))
lr_multiplier = float(dbutils.widgets.get("lr-multiplier"))
scale_lr_layer = dbutils.widgets.get("scale-lr-layer")
micro_batch_size = int(dbutils.widgets.get("micro-batch-size"))
precision = dbutils.widgets.get("precision")

os.makedirs(work_dir + "/data")
os.makedirs(work_dir + "/ft_weights")