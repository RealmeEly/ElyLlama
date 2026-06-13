#include "model/model.h"

void ModelFactory::init() {
  //
}

// automatic initialization
static class ModelFactoryInitializer {
public:
  ModelFactoryInitializer() {
    ModelFactory::init();
  }
} global_model_factory_initializer;
