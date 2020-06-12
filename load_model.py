from keras.models import load_model
model = load_model('model/renew.en.msd.weights.best.hdf5')
model.save_weights('en_msd_weights.hdf5')
