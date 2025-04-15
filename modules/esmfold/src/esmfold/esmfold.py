import mlflow
import transformers
from typing import List

class ESMFoldPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        CACHE_DIR = context.artifacts["cache"]

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/esmfold_v1", cache_dir=CACHE_DIR
        )
        self.model = transformers.EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True, cache_dir=CACHE_DIR
        )

        self.model = self.model.cuda()
        self.model.esm = self.model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True

    def _post_process(self, outputs):
        final_atom_positions = (
            transformers.models.esm.openfold_utils.feats.atom14_to_atom37(
                outputs["positions"][-1], outputs
            )
        )
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = transformers.models.esm.openfold_utils.protein.Protein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=(
                    outputs["chain_index"][i] if "chain_index" in outputs else None
                ),
            )
            pdbs.append(transformers.models.esm.openfold_utils.protein.to_pdb(pred))
        return pdbs

    def predict(self, context, model_input: List[str], params=None) -> List[str]:
        tokenized_input = self.tokenizer(
            model_input, return_tensors="pt", add_special_tokens=False, padding=True
        )["input_ids"]
        tokenized_input = tokenized_input.cuda()
        with torch.no_grad():
            output = self.model(tokenized_input)
        pdbs = self._post_process(output)
        return pdbs