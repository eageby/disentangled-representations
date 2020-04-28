from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt

import numpy as np
import tensorflow as tf

class Latent(QtWidgets.QWidget):
    updated_value = QtCore.Signal(float)
    def __init__(self, idx, value, min_=0, max_=1, *args, **kwargs):
        super(Latent, self).__init__()
        self.min_ = min_
        self.max_ = max_
        self.value = value
        self.idx = idx

        layout = QtWidgets.QVBoxLayout()
        self.setMinimumWidth(100)
        self.setMaximumWidth(500)
        self.setMaximumHeight(100)

        self.label = QtWidgets.QLabel(text='Latent Variable {}'.format(self.idx))
        self.label.setTextInteractionFlags(Qt.TextEditorInteraction)
        layout.addWidget(self.label)

        self.slider = QtWidgets.QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(0.01)
        self.slider.setMinimum(100*min_)
        self.slider.setMaximum(100*max_)
 
        layout.addWidget(self.slider)

        self.values = QtWidgets.QHBoxLayout()
        self.values.setObjectName(u"values")
        self.min = QtWidgets.QLabel(self)
        self.min.setObjectName(u"min")
        self.min.setText("{:.2f}".format(min_))

        self.values.addWidget(self.min, 0, Qt.AlignLeft|Qt.AlignTop)

        self.current = QtWidgets.QDoubleSpinBox(self)
        self.current.setObjectName(u"current")
        self.current.setDecimals(2)
        self.current.setSingleStep(0.1)
        self.current.setMaximum(self.min_)
        self.current.setMaximum(self.max_)
        self.current.setMaximumSize(QtCore.QSize(60, 30))


        self.values.addWidget(self.current)

        self.max = QtWidgets.QLabel(self)
        self.max.setObjectName(u"max")
        self.max.setText("{:.2f}".format(max_))

        self.values.addWidget(self.max, 0, Qt.AlignRight|Qt.AlignTop)
        layout.addLayout(self.values)

        self.set(self.value)

        self.setLayout(layout)

        self.slider.valueChanged.connect(self.update_current)
        self.current.valueChanged.connect(self.update_slider)

        self.updated_value.connect(self.set)

    def set(self, value):
        self.value = value                
        self.slider.setValue(100 *value)
        self.current.setValue(value)

    def update_current(self, value):
        self.current.setValue(value/100)

    def update_slider(self, value):
        self.slider.setValue(100 * value)
        self.updated_value.emit(value)

class LatentVariables(QtWidgets.QWidget):
    representationChanged = QtCore.Signal(object)
    def __init__(self, initial, min_, max_, indices,n_variables=2,*args, **kwargs):
        super(LatentVariables, self).__init__(*args, **kwargs)
        self.setMaximumWidth(500)

        self.initial = np.asarray(initial)

        self.representation = self.initial.copy() 
        self.indices = indices

        self.max_ = max_
        self.min_ = min_
        self.n_max = len(initial)


        scroll = QtWidgets.QScrollArea()
        layout = QtWidgets.QVBoxLayout(self)
        self.topSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        layout.addItem(self.topSpacer)

        self.variables = []
        self.variable_layout = QtWidgets.QVBoxLayout()
        self.set_variables(n_variables) 

        layout.addLayout(self.variable_layout)

        self.button = QtWidgets.QPushButton(text='Reset')
        self.button.setMaximumWidth(100)
        self.button.clicked.connect(self.reset)
        layout.addWidget(self.button, alignment=Qt.AlignHCenter)

        # layout.addItem(self.botSpacer)
        self.setLayout(layout)
        

    def set_sample(self, s):
        self.sample = s

    def reset(self):
        self.set(self.initial)

    def add_variable(self):
        i = len(self.variables)
        if i < self.n_max:
            latent = Latent(self.indices[i], float(self.initial[self.indices[i]]), self.min_[self.indices[i]], self.max_[self.indices[i]])
            self.variables.append(latent)
            self.variable_layout.addWidget(latent) 
            latent.updated_value.connect(self.update)

    def remove_variable(self):
        i = len(self.variables)
        if i > 1:
            latent = self.variables.pop()
            self.variable_layout.removeWidget(latent)
            latent.setParent(None)

    def set_variables(self, number):
        i = len(self.variables)
        while i < number:
            self.add_variable()
            i += 1
        while i > number:
            self.remove_variable()
            i -= 1

    def update(self):
        values = np.asarray([v.value for v in self.variables])
        self.representation[self.indices[:len(self.variables)]] = values
        self.representationChanged.emit(self.representation)

    def set(self, representation):
        self.initial = representation.copy()
        self.representation = representation
        for i, v in enumerate(self.variables):
            v.set(self.initial[self.indices[i]])


class Options(QtWidgets.QWidget):
    def __init__(self, model, dataset, latents, batch_size, *args, **kwargs):
        super(Options, self).__init__(*args, **kwargs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(sizePolicy)

        self.layout = QtWidgets.QGridLayout(self)

        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)

        self.sampleBox = QtWidgets.QSpinBox(self)
        self.sampleBox.setMinimum(0)
        self.sampleBox.setMaximum(batch_size)

        self.layout.addWidget(self.sampleBox, 5, 4, 1, 1)

        self.latentLabel = QtWidgets.QLabel(self, text='Latent Variables')
        self.latentLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.latentLabel.setFont(font)
        self.layout.addWidget(self.latentLabel, 0, 3, 1, 1)

        self.datasetLabel = QtWidgets.QLabel(self, text='Dataset')
        self.datasetLabel.setFont(font)
        self.layout.addWidget(self.datasetLabel, 0, 2, 1, 1)

        self.modelLabel = QtWidgets.QLabel(self, text='Model')
        self.modelLabel.setFont(font)
        self.layout.addWidget(self.modelLabel, 0, 1, 1, 1)

        self.leftSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(self.leftSpacer, 5, 0, 1, 1)

        self.rightSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.layout.addItem(self.rightSpacer, 5, 5, 1, 1)

        self.sampleLabel = QtWidgets.QLabel(self, text='Sample')
        self.sampleLabel.setFont(font)
        self.layout.addWidget(self.sampleLabel, 0, 4, 1, 1)

        self.latentVariablesBox = QtWidgets.QSpinBox(self)
        self.latentVariablesBox.setMaximumSize(QtCore.QSize(80, 16777215))
        self.latentVariablesBox.setMinimum(1)
        self.layout.addWidget(self.latentVariablesBox, 5, 3, 1, 1)

        self.dataset = QtWidgets.QLabel(self, text=dataset)
        self.layout.addWidget(self.dataset, 5, 2, 1, 1)

        self.model = QtWidgets.QLabel(self, text=model)
        self.layout.addWidget(self.model, 5, 1, 1, 1)
        self.midSpacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.layout.addItem(self.midSpacer, 5, 3, 1, 1)


        self.rightSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.leftSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.layout.addItem(self.leftSpacer,5,0 ,1,1)
        self.layout.addItem(self.rightSpacer,5,6 ,1,1)
        
        self.shuffleButton = QtWidgets.QPushButton(text='Shuffle')
        self.shuffleButton.setMaximumWidth(80)
        self.layout.addWidget(self.shuffleButton, 5, 5, 1, 1)
        self.shuffleButton.clicked.connect(self.random_sample)

        self.set_max_latent_variables(latents)

    def set_max_latent_variables(self, max_):
        self.latentVariablesBox.setMaximum(max_)

    def random_sample(self):
        sample = np.random.randint(0, self.sampleBox.maximum())
        self.sampleBox.setValue(sample)

class Image(QtWidgets.QLabel):
    def __init__(self):
        super(Image, self).__init__()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.setSizePolicy(sizePolicy)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(64, 64))
        self.setMaximumSize(QtCore.QSize(2048, 2048))
        self.setScaledContents(True)

    def paintEvent(self, event):
        # size = self.size()
        # painter = QtGui.QPainter(self)
        # point = QtCore.QPoint(0,0)
        # # start painting the label from left upper corner
        # point.setX((size.width() - scaledPix.width())/2)
        # point.setY((size.height() - scaledPix.height())/2)
        # painter.drawPixmap(point, scaledPix)
        self.setFixedHeight(self.size().width())
        super(Image, self).paintEvent(event)

class Visualize(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(Visualize, self).__init__()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.setSizePolicy(sizePolicy)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.rightSpacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(self.rightSpacer, 2, 4, 1, 1)

        self.leftSpacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)

        self.gridLayout.addItem(self.leftSpacer, 2, 2, 1, 1)

        self.outputLabel = QtWidgets.QLabel(self,text='Output')
        self.outputLabel.setObjectName(u"outputLabel")
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.outputLabel.setFont(font)
        self.outputLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.outputLabel, 0, 5, 1, 1)

        self.representationLabel = QtWidgets.QLabel(self,text='Representation')
        self.representationLabel.setObjectName(u"representationLabel")
        self.representationLabel.setFont(font)
        self.representationLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.representationLabel, 0, 3, 1, 1)

        self.representationLine = QtWidgets.QFrame(self)
        self.representationLine.setObjectName(u"representationLine")
        self.representationLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.representationLine.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.gridLayout.addWidget(self.representationLine, 1, 3, 1, 1)

        self.outputLine = QtWidgets.QFrame(self)
        self.outputLine.setObjectName(u"outputLine")
        self.outputLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.outputLine.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.gridLayout.addWidget(self.outputLine, 1, 5, 1, 1)

        self.outputImage = Image()
        self.gridLayout.addWidget(self.outputImage, 2, 5, 2, 1)

        self.inputLabel = QtWidgets.QLabel(self, text='Input')
        self.inputLabel.setObjectName(u"inputLabel")
        self.inputLabel.setFont(font)
        self.inputLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.inputLabel, 0, 1, 1, 1)

        self.inputImage = Image()
        self.gridLayout.addWidget(self.inputImage, 2, 1, 1, 1)

        self.inputLine = QtWidgets.QFrame(self)
        self.inputLine.setObjectName(u"inputLine")
        sizePolicy1 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.inputLine.sizePolicy().hasHeightForWidth())
        self.inputLine.setSizePolicy(sizePolicy1)
        self.inputLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.inputLine.setLineWidth(1)
        self.inputLine.setFrameShape(QtWidgets.QFrame.HLine)

        self.gridLayout.addWidget(self.inputLine, 1, 1, 1, 1)
 
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setMaximumWidth(500)
        
        self.latent_variables = LatentVariables(**kwargs)
        self.scroll.setWidget(self.latent_variables)
        self.gridLayout.addWidget(self.scroll, 2, 3, 1, 1)

    def convert_image(self, image):
        image = np.asarray(255 * image, dtype=np.uint8)
        qimage = QtGui.QImage(image, image.shape[0],image.shape[1], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap(qimage)

    def set_input(self, input_):
        self.inputImage.setPixmap(self.convert_image(input_))

    def set_output(self, output):
        self.outputImage.setPixmap(self.convert_image(output))

class MainWindow(QtWidgets.QWidget):
    def __init__(self, model, data, model_name='', dataset_name='', **kwargs):
        super(MainWindow, self).__init__()
        self.model = model
        self.data = np.asarray(data)

        layout = QtWidgets.QVBoxLayout(self)

        self.options = Options(model_name, dataset_name, 32, 128)
        layout.addWidget(self.options)
        
        representation = np.asarray(self.model.encode(self.data)[0])

        min_ = np.asarray(tf.math.reduce_min(representation, axis=0))
        max_ = np.asarray(tf.math.reduce_max(representation, axis=0))
        var = tf.math.reduce_variance(representation, axis=0)
        idx = np.argsort(var)[::-1]

        self.visualize = Visualize(initial=representation[0], min_=min_, max_=max_, indices=idx)
        self.options.set_max_latent_variables(representation.shape[-1])
        self.options.latentVariablesBox.valueChanged.connect(self.visualize.latent_variables.set_variables)
        self.options.latentVariablesBox.valueChanged.connect(self.size)
        self.options.set_max_latent_variables(self.visualize.latent_variables.n_max)
        self.options.latentVariablesBox.setValue(len(self.visualize.latent_variables.variables))

        layout.addWidget(self.visualize)

        self.setLayout(layout)
        self.resize(QtCore.QSize(1100, 500))
        
        self.update_sample(self.options.sampleBox.value())
        
        self.options.sampleBox.valueChanged.connect(self.update_sample)
        self.visualize.latent_variables.representationChanged.connect(self.update_representation)

    def update_representation(self, representation):
        self.representation = np.asarray(representation)
        self.output = self.model.decode(self.representation[None])[0][0]
        self.visualize.set_output(self.output)

    def update_sample(self, sample):
        self.sample = sample
        self.visualize.set_input(self.data[self.sample])

        self.representation = np.asarray(self.model.encode(self.data[None, self.sample])[0])[0].copy()
        self.visualize.latent_variables.set(self.representation)

        self.output = np.asarray(self.model.decode(self.representation[None])[0])[0]
        self.visualize.set_output(self.output)

def main(model, dataset, batch_size=128, shuffle=False, **kwargs):
    app = QtWidgets.QApplication([])
    data = dataset.pipeline(batch_size)
    
    batch = data.as_numpy_iterator().next()
    main = MainWindow(model, batch, **kwargs)

    main.show()
    app.exec_()
