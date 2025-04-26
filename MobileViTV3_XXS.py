from functools import partial
import keras
class MobileViT_V3XXS():

  def build_model(input_shape,include_top=True,num_classes=5,expansion_factor=2,patch_size=4,**kwargs):

    conv_block = partial(keras.layers.Conv2D,filters=16,kernel_size=3,strides=2, activation=keras.activations.swish,padding="same")
    


    inputs = keras.Input(shape=input_shape)


    x = conv_block( filters=16)(inputs) #  conv_block( x,filters=16)
    x = MobileViT_V3XXS.inverted_residual_block(
       x=x, expanded_channels=16 *expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )

    # First MV2 -> MobileViT block.
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=64, strides=2
    )
    x = MobileViT_V3XXS.mobilevit_block_v3(x=x,conv_block=conv_block,patch_size=patch_size, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=80, strides=2
    )
    x = MobileViT_V3XXS.mobilevit_block_v3(x,conv_block=conv_block,patch_size=patch_size, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = MobileViT_V3XXS.inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=96, strides=2
    )
    x = MobileViT_V3XXS.mobilevit_block_v3(x,conv_block=conv_block,patch_size=patch_size, num_blocks=3, projection_dim=96)
    x = conv_block( filters=320, kernel_size=1, strides=1)(x)

    # Classification head.



    if include_top:
      x = keras.layers.GlobalAvgPool2D()(x)
      outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    else:
      outputs = x


    return keras.Model(inputs, outputs,name='ViT3XXS')




#   def get_model(self):
#     return self.built_model

  def correct_pad(inputs, kernel_size):
      img_dim = 2 if keras.backend.image_data_format() == "channels_first" else 1
      input_size = inputs.shape[img_dim : (img_dim + 2)]
      if isinstance(kernel_size, int):
          kernel_size = (kernel_size, kernel_size)
      if input_size[0] is None:
          adjust = (1, 1)
      else:
          adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
      correct = (kernel_size[0] // 2, kernel_size[1] // 2)
      return (
          (correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]),
      )


  # Reference: https://git.io/JKgtC


  def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
      m = keras.layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
      m = keras.layers.BatchNormalization()(m)
      m = keras.activations.swish(m)

      if strides == 2:
          m = keras.layers.ZeroPadding2D(padding=MobileViT_V3XXS.correct_pad(m, 3))(m)
      m = keras.layers.DepthwiseConv2D(
          3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
      )(m)
      m = keras.layers.BatchNormalization()(m)
      m = keras.activations.swish(m)

      m = keras.layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
      m = keras.layers.BatchNormalization()(m)

      if keras.ops.equal(x.shape[-1], output_channels) and strides == 1:
          return keras.layers.Add()([m, x])
      return m


  # Reference:
  # https://keras.io/examples/vision/image_classification_with_vision_transformer/


  def mlp(x, hidden_units, dropout_rate):
      for units in hidden_units:
          x = keras.layers.Dense(units, activation=keras.activations.swish)(x)
          x = keras.layers.Dropout(dropout_rate)(x)
      return x


  def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
      for _ in range(transformer_layers):
          # Layer normalization 1.
          x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
          # Create a multi-head attention layer.
          attention_output = keras.layers.MultiHeadAttention(
              num_heads=num_heads, key_dim=projection_dim, dropout=0.1
          )(x1, x1)
          # Skip connection 1.
          x2 = keras.layers.Add()([attention_output, x])
          # Layer normalization 2.
          x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
          # MLP.
          x3 = MobileViT_V3XXS.mlp(
              x3,
              hidden_units=[x.shape[-1] * 2, x.shape[-1]],
              dropout_rate=0.1,
          )
          # Skip connection 2.
          x = keras.layers.Add()([x3, x2])

      return x

  def mobilevit_block_v3(x,conv_block,patch_size,num_blocks, projection_dim, strides=1):
      local_features = keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=strides, activation=keras.activations.swish,padding="same")(x)
      local_features =  conv_block(
          filters=projection_dim, kernel_size=1, strides=strides
      )(local_features)

      skip2 = local_features


      # Unfold into patches and then pass through Transformers.
      num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)

      non_overlapping_patches = keras.layers.Reshape((patch_size, num_patches, projection_dim))(
          local_features
      )

      global_features = MobileViT_V3XXS.transformer_block(
          non_overlapping_patches, num_blocks, projection_dim
      )

      # Fold into conv-like feature-maps.
      folded_feature_map = keras.layers.Reshape((*local_features.shape[1:-1], projection_dim))(
          global_features
      )

      # Apply point-wise conv -> concatenate with the input features.
      folded_feature_map =  conv_block(
          filters=skip2.shape[-1], kernel_size=1, strides=strides
      )(folded_feature_map)

      local_global_features = keras.layers.Concatenate(axis=-1)([skip2, folded_feature_map])

      # Fuse the local and global features using a convoluion layer.
      local_global_features =  conv_block( filters=projection_dim, strides=strides)(local_global_features)

      return local_global_features+x


        
