from helper import *


def process_dataset_triples(dataset:str, save_dir:str, example_dir:str, lbl_triples:List, process_data_flag:Optional[bool]=True, 
                            display:Optional[bool]=True):
    if process_data_flag:
        # Format facts

        lbl_fact_mat, fact_ids, triples = process_metadata(lbl_triples, prefix="fact")
        fact_txt = [" ".join(o) for o in triples]
        joblib.dump(triples, f"{save_dir}/raw_data/{dataset}-hipporag-triples.joblib")

        sp.save_npz(f"{save_dir}/{dataset}-hipporag-fact_lbl_X_Y.npz", lbl_fact_mat)
        save_raw_file(f"{save_dir}/raw_data/{dataset}-hipporag-fact.raw.csv", fact_ids, fact_txt)

        # Format entities

        lbl_entities = [chain(*[[k[0], k[2]] for k in o]) for o in lbl_triples]
        lbl_entity_mat, entity_ids, entity_txt = process_metadata(lbl_entities, prefix="entity")

        sp.save_npz(f"{save_dir}/{dataset}-hipporag-entity_lbl_X_Y.npz", lbl_entity_mat)
        save_raw_file(f"{save_dir}/raw_data/{dataset}-hipporag-entity.raw.csv", entity_ids, entity_txt)
        
        ent_ent = create_entity_graph(entity_txt, lbl_triples)
        sp.save_npz(f"{save_dir}/{dataset}-hipporag-entity_{dataset}-hipporag-entity_X_Y.npz", ent_ent)

        fact_ent = create_fact_entity_graph(triples, entity_txt, lbl_triples)
        sp.save_npz(f"{save_dir}/{dataset}-hipporag-entity_{dataset}-hipporag-fact_X_Y.npz", fact_ent)

    if display:
        get_fact_examples(example_dir, save_dir, fact_txt, entity_txt, lbl_fact_mat, lbl_entity_mat)
        get_entity_examples(example_dir, entity_txt, ent_ent)


def load_generations(data_dir:str, fname:str):
    data_dir = "/data/datasets/beir/metadata/raw/05-beir_hipporag_metadata/hotpotqa_label_and_entity_remaining_samples-outputs/"
    fname = "UHRS_Task"

    output_file = f"{data_dir}/{fname}.raw.tsv"
    if os.path.exists(output_file):
        df = pd.read_table(output_file)
    else:
        all_files = [f"{data_dir}/{fname}_{i}.tsv" for i in range(1, 60)]
        df = pd.concat([pd.read_table(f) for f in all_files], axis=0)
    return df["id"].tolist(), df["raw_model_response"].tolist()


if __name__ == "__main__":

    display, process_data_flag = True, True

    output_dir = "/data/datasets/beir/metadata/raw/05-beir_hipporag_metadata/"
    
    dataset = "hotpotqa"

    # Load triples
    if process_data_flag:

        fname = f"{output_dir}/{dataset}_label_ids_and_triples.joblib"
        if os.path.exists(fname):
            lbl_ids, lbl_triples = joblib.load(fname)
        else:
            lbl_ids, lbl_triples = load_generations()
            lbl_triples = preprocess_triples(lbl_triples)
            joblib.dump((lbl_ids, lbl_triples), fname)

        assert len(lbl_ids) == len(lbl_triples)

        dset2idx = dict()
        for i, ids in enumerate(lbl_ids):
            dset2idx.setdefault(ids.split("_", maxsplit=1)[0], []).append(i)

        # Save triple information for each dataset

        save_dir = f"/data/datasets/beir/{dataset}/XC/"
        example_dir = f"/data/datasets/beir/{dataset}/examples/"
        os.makedirs(example_dir, exist_ok=True)

        dset_tag = dataset.replace("/", "-")

        ids, triples = [lbl_ids[i].split("_", maxsplit=1)[1] for i in dset2idx[dset_tag]], [lbl_triples[i] for i in dset2idx[dset_tag]]
        ids2triples = {k:v for k,v in zip(ids, triples)}

        ids, txt = load_raw_file(f"{save_dir}/raw_data/label.raw.csv")
        ids = [str(i) for i in ids]
        
        triples = [ids2triples.get(i, []) for i in ids]
        joblib.dump(triples, f"{save_dir}/raw_data/label_{dataset}-hipporag-triples.joblib")

        process_dataset_triples(dataset, save_dir, example_dir, triples, process_data_flag=process_data_flag, display=display)

    elif display:

        save_dir = f"/data/datasets/beir/{dataset}/XC/"
        example_dir = f"/data/datasets/beir/{dataset}/examples/"

        fact_ids, fact_txt = load_raw_file(f"{save_dir}/raw_data/{dataset}-hipporag-fact.raw.csv")
        entity_ids, entity_txt = load_raw_file(f"{save_dir}/raw_data/{dataset}-hipporag-entity.raw.csv")

        lbl_entity_mat = sp.load_npz(f"{save_dir}/{dataset}-hipporag-entity_lbl_X_Y.npz")
        lbl_fact_mat = sp.load_npz(f"{save_dir}/{dataset}-hipporag-fact_lbl_X_Y.npz")
        ent_ent = sp.load_npz(f"{save_dir}/{dataset}-hipporag-entity_{dataset}-hipporag-entity_X_Y.npz")

        get_fact_examples(example_dir, save_dir, fact_txt, entity_txt, lbl_fact_mat, lbl_entity_mat)
        get_entity_examples(example_dir, entity_txt, ent_ent)

