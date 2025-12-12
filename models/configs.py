import ml_collections

def get_vl16_config():
    """Returns the ViT-L/16 configuration."""
    attention_dropout = 0.1
    drop_out = 0.1
    config = ml_collections.ConfigDict()
    config.hidden_size = 1024  # d=1024
    config.classifier = 'token'
    config.representation_size = None

    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 4096  # typically 4x hidden_size
    config.spatial.transformer.num_heads = 16  # NH=16
    config.spatial.transformer.num_layers = 24  # L=24
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 4096
    config.temporal.transformer.num_heads = 16
    config.temporal.transformer.num_layers = 4  # keeping temporal layers smaller
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size

    return config


def get_vb16_config():
    """Returns the ViT-B/16 configuration."""
    attention_dropout = 0.1
    drop_out = 0.1
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.classifier = 'token'
    config.representation_size = None


    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 12
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size



    return config


def get_vb16_config_small():
    """Returns the ViT-B/16 Small 6 layers configuration."""
    attention_dropout = 0.1
    drop_out = 0.1
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.classifier = 'token'
    config.representation_size = None


    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 6
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size

    return config


def get_epic_kitchens_config():
    """Returns the ViT-L/16 configuration for Epic Kitchens dataset.
    
    Epic Kitchens uses multi-head classification with:
    - 300 noun classes
    - 97 verb classes
    Total: 397 classes
    
    Note: Order matches scenic-vivit config (noun first, then verb)
    """
    attention_dropout = 0.0  # Match scenic-vivit settings
    drop_out = 0.0
    config = ml_collections.ConfigDict()
    config.hidden_size = 1024  # ViT-L uses 1024
    config.classifier = 'token'
    config.representation_size = None
    config.label_smoothing = 0.2  # Epic Kitchens uses label smoothing
    
    # Multi-head classification settings (order matches scenic-vivit: noun, verb)
    config.class_splits = [300, 97]  # [noun, verb]
    config.split_names = ['noun', 'verb']
    config.num_classes = sum(config.class_splits)  # 397 total

    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 4096
    config.spatial.transformer.num_heads = 16
    config.spatial.transformer.num_layers = 24
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 4096
    config.temporal.transformer.num_heads = 16
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size
    
    # Stochastic depth (droplayer)
    config.stochastic_droplayer_rate = 0.2

    return config


def get_vb16_epic_kitchens_config():
    """Returns the ViT-B/16 configuration for Epic Kitchens dataset.
    
    Smaller model version for Epic Kitchens multi-head classification.
    Note: Order matches scenic-vivit config (noun first, then verb)
    """
    attention_dropout = 0.0
    drop_out = 0.0
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.classifier = 'token'
    config.representation_size = None
    config.label_smoothing = 0.2
    
    # Multi-head classification settings (order matches scenic-vivit: noun, verb)
    config.class_splits = [300, 97]  # [noun, verb]
    config.split_names = ['noun', 'verb']
    config.num_classes = sum(config.class_splits)

    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 12
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size
    
    config.stochastic_droplayer_rate = 0.1

    return config