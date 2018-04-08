from pkg.pkg import PKG

PKG.evaluate_embeddings(embedding_path="temp/embedding_TransE/embedding.vec.json",
                        openke_dir="temp/openke/",
                        user_info_path="temp/sample_data/user_id_imei_birth_gender.txt")
