"""
Usage:
1. training
python squeezenet_demo.py --action='train'\
    -p /home/db/train -v /home/db/validation
2. prediction
python squeezenet_demo.py --action='predice'\
    -p /db/Roasted-Broccoli-Pasta-Recipe-5-683x1024.jpg
"""
import time
import argparse
from simdat.core import dp_tools
from simdat.core import keras_models as km
from simdat.core import tools
from keras.optimizers import Adam
from keras.optimizers import SGD

dp = dp_tools.DP()
tl = tools.DATA()


def main():
    parser = argparse.ArgumentParser(
        description="SqueezeNet example."
        )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest='batchsize',
        help="Size of the mini batch. Default: 32."
        )
    parser.add_argument(
        "--action", type=str, default='train',
        help="Action to be performed, train/predict"
        )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate of SGD, default 0.001."
        )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Epsilon of Adam epsilon, default 1e-8."
        )
    parser.add_argument(
        "-p", "--path", type=str, default='.', required=True,
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "-v", "--val-path", type=str, default='.',
        dest='valpath', help="Path where the val images are. Default: $PWD."
        )
    parser.add_argument(
        "--img-width", type=int, default=224, dest='width',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-height", type=int, default=224, dest='height',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Channels of the images, default: 3."
        )

    args = parser.parse_args()
    sgd = SGD(lr=args.lr, decay=0.0002, momentum=0.9)

    t0 = time.time()
    if args.action == 'train':
        train_generator = dp.train_data_generator(
            args.path, args.width, args.height)
        validation_generator = dp.val_data_generator(
            args.valpath, args.width, args.height)

        classes = train_generator.class_indices
        nb_train_samples = train_generator.nb_sample
        nb_val_samples = validation_generator.nb_sample
        print("[squeezenet_demo] N training samples: %i " % nb_train_samples)
        print("[squeezenet_demo] N validation samples: %i " % nb_val_samples)
        nb_class = train_generator.nb_class
        print('[squeezenet_demo] Total classes are %i' % nb_class)

        t0 = tl.print_time(t0, 'initialize data')
        model = km.SqueezeNet(
            nb_class, inputs=(args.channels, args.height, args.width))
        dp.visualize_model(model)
        t0 = tl.print_time(t0, 'build the model')

        model.compile(
            optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])
        t0 = tl.print_time(t0, 'compile model')

        model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=args.epochs,
            validation_data=validation_generator,
            nb_val_samples=nb_val_samples)
        t0 = tl.print_time(t0, 'train model')

        model.save_weights('./weights.h5', overwrite=True)
        model_parms = {'nb_class': nb_class,
                       'nb_train_samples': nb_train_samples,
                       'nb_val_samples': nb_val_samples,
                       'classes': classes,
                       'channels': args.channels,
                       'height': args.height,
                       'width': args.width}
        tl.write_json(model_parms, fname='./model_parms.json')
        t0 = tl.print_time(t0, 'save model')

    elif args.action == 'predict':
        _parms = tl.parse_json('./model_parms.json')
        model = km.SqueezeNet(
            _parms['nb_class'],
            inputs=(_parms['channels'], _parms['height'], _parms['width']),
            weights_path='./weights.h5')
        dp.visualize_model(model)
        model.compile(
            optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])

        X_test, Y_test, classes, F = dp.prepare_data_test(
            args.path, args.width, args.height)
        t0 = tl.print_time(t0, 'prepare data')

        outputs = []
        results = model.predict(
            X_test, batch_size=args.batchsize, verbose=1)
        classes = _parms['classes']
        for i in range(0, len(F)):
            _cls = results[i].argmax()
            max_prob = results[i][_cls]
            outputs.append({'input': F[i], 'max_probability': max_prob})
            cls = [key for key in classes if classes[key] == _cls][0]
            outputs[-1]['class'] = cls
            print('[squeezenet_demo] %s: %s (%.2f)' % (F[i], cls, max_prob))
        t0 = tl.print_time(t0, 'predict')

if __name__ == '__main__':
    main()
