from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.gfile import GFile
from tensorflow import GraphDef, Graph, import_graph_def, Session

model_dir = "facenet.pb"
output_file = "./deploy_model/1"

def main():
    with GFile(model_dir, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())

    with Session() as sess:
        # Then, we import the graph_def into a new Graph and returns it
        with Graph().as_default() as graph:
            import_graph_def(graph_def, name='')
            signature = predict_signature_def(
                inputs={'input': graph.get_tensor_by_name('image_batch:0'),
                        'phase_train': graph.get_tensor_by_name('phase_train:0')},
                outputs={'embeddings': graph.get_tensor_by_name('embeddings:0')}
            )
            builder = saved_model_builder.SavedModelBuilder(output_file)
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={'serving_default': signature}
            )
            builder.save()


if __name__ == '__main__':
    main()