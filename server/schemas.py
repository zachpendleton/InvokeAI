# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from marshmallow import Schema, SchemaOpts, fields, INCLUDE
from marshmallow.fields import Field
from marshmallow.validate import OneOf, Range
from marshmallow_oneofschema import OneOfSchema

from ldm.dream.generator.embiggen import Embiggen

class ImageField(Field):
  """Used to mark image inputs/outputs. Only usable with links."""
  def __init__(self, *args, **kwargs):
    kwargs.setdefault('load_default', None)
    kwargs.setdefault('dump_default', None)
    kwargs.setdefault('allow_none', True)
    super().__init__(*args, **kwargs)


class ProcessorSchemaOpts(SchemaOpts):
  """Adds outputs to Meta options to enable link validation."""

  def __init__(self, meta, **kwargs):
    SchemaOpts.__init__(self, meta, **kwargs)
    self.outputs = getattr(meta, 'outputs', {})
    self.type = getattr(meta, 'type', None)


class ProcessorSchema(Schema):
  OPTIONS_CLASS = ProcessorSchemaOpts

  id = fields.String(required=True)


# Processors
# Only mark required=True on fields that can't be used as links

class ImageSchema(ProcessorSchema):
  """Image loader"""
  class Meta:
    type = 'image'
    outputs = {
      'image': ImageField()
    }

  image = fields.String()


class GFPGANSchema(ProcessorSchema):
  """Face restoration"""
  class Meta:
    type = 'gfpgan'
    outputs = {
      'image': ImageField()
    }

  image = ImageField()
  gfpgan_strength = fields.Float(required=True, validate=Range(0.0, 1.0, min_inclusive=False, max_inclusive=True))


class UpscaleSchema(ProcessorSchema):
  """Upscale"""
  class Meta:
    type = 'upscale'
    outputs = {
      'image': ImageField()
    }

  image = ImageField()
  level = fields.Integer(required=True, validate=OneOf([2, 4]))
  strength = fields.Float(required=True, validate=Range(0.0, 1.0, min_inclusive=False, max_inclusive=True))


# # TODO: Fill this out
# class EmbiggenSchema(ProcessorSchema):
#   """Embiggen"""
#   embiggen = fields.Raw()
#   embiggen_tiles = fields.Raw()


class GenerateSchema(ProcessorSchema):
  """txt2img"""
  class Meta:
    type = 'generate'
    outputs = {
      'image': ImageField()
    }
    # TODO: output intermediates? That doesn't seem quite right, since they couldn't be used

  prompt = fields.String(required=True)
  seed = fields.Integer(load_default=0) # 0 is random
  steps = fields.Integer(load_default=10)
  width = fields.Integer(load_default=512)
  height = fields.Integer(load_default=512)
  cfg_scale = fields.Float(load_default=7.5)
  sampler_name = fields.String(load_default='klms',
    validate=OneOf(['ddim','plms','k_lms','k_dpm_2','k_dpm_2_a','k_euler','k_euler_a','k_heun']))
  seamless = fields.Boolean(load_default=False)
  model = fields.String() # currently unused
  embeddings = fields.Raw() # currently unused
  progress_images = fields.Boolean(load_default='false')


class ImageToImageSchema(GenerateSchema):
  """img2img, runs txt2img with a weighted initial image"""
  class Meta(GenerateSchema.Meta):
    type = 'img2img'

  image = ImageField()
  strength = fields.Float(required=True, validate=Range(0.0, 1.0, min_inclusive=False, max_inclusive=True))
  fit = fields.Boolean(required=True)


class ProcessorsSchema(OneOfSchema):
  """OneOfSchema that can load all processors if their 'type' matches"""
  @staticmethod
  def __all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in ProcessorsSchema.__all_subclasses(c)])

  def __init__(self):
    # Define this in init instead of class so it catches processor schemas defined outside this file
    self.type_schemas = dict(map(lambda s: (s.Meta.type, s), ProcessorsSchema.__all_subclasses(ProcessorSchema)))


# Set this here so it can use the static method
# TODO: this won't support adding processors from outside this file - will need to figure that out


class DreamMapSchema(Schema):
  """The map for processing"""
  nodes = fields.List(fields.Nested(ProcessorSchema))
  #links = fields.List(fields.Link(LINKTYPE?))

  # validate_schema (validate all links and their types)


# NOTE: OLD CLASS BELOW HERE
# class DreamBaseSchema(Schema):
#   # Allow unknown data to be deserialized (until code stabilizes)
#   class Meta:
#     unknown = INCLUDE
  
#   # Id
#   id = fields.String()

#   # Metadata
#   time = fields.Integer()

#   # Initial Image
#   image = fields.Nested(ImageSchema)

#   # Img2Img
#   img2img = fields.Nested(ImageToImageSchema)

#   # Generation
#   generate = fields.Nested(GenerateSchema)

#   # GFPGAN
#   gfpgan = fields.Nested(GFPGANSchema)

#   # Upscale
#   upscale = fields.Nested(UpscaleSchema)

#   # Embiggen
#   embiggen = fields.Nested(EmbiggenSchema)
