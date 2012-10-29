from traits.api import  Str, Int, List, Button, HasTraits, Bool
from traitsui.api import View, Item, CheckListEditor, Group, Handler, Action, BooleanEditor, DirectoryEditor
from .experiment import ExpModel, ExpCorpus, ExpClassifier
from .misc import SetCorpus
from glimpse import backends, models

class TempStr(HasTraits):
  """Provides UI for a string input"""
  value = Str
  
class TempDir(HasTraits):
  """Provides directory input"""
  value = Str (editor = DirectoryEditor())
  
class UI(HasTraits):
  
  ListModel = List  
  ListCorpus = List
  ListClassifier = List 
  
  ListModelView = List(editor = CheckListEditor (name = "ListModel"))
  ListCorpusView = List( editor = CheckListEditor (name = 'ListCorpus'))
  ListClassifierView = List( editor = CheckListEditor (name = 'ListClassifier'))
  Corpus_Dir = Str
  Result = Str
  
  AddModelButton = Button('AddModel')
  AddCorpusButton = Button ('AddCorpus')
  AddClassifierButton = Button ('Classifier')
  
  MakePrototypesButton = Button ('Make prototypes')
  
  ListPrototypeSources = List (editor = 
                              CheckListEditor(values = [('MakeHistogramRandomS2Prototypes','Histogram Random'),
                                                        ('MakeNormalRandomS2Prototypes', 'Normal Random'),
                                                        ('MakeShuffledRandomS2Prototypes', 'Shuffled Random'),
                                                        ('MakeUniformRandomS2Prototypes', 'Uniform Random'),
                                                        ('ImprintS2Prototypes', 'Imprint from Corpus')]))
  
  ComputeFeatureButton = Button ('Compute Feature for corpus')
  TrainSvmButton = Button ('Train Svm')
  TestSvmButton = Button ('Test Svm')
  
  
  CurrentModelName = Str
  CurrentModelPrototypes =  Str
  CurrentModelLayer = Str
  
  def _ListModelView_changed(self):
    model = self.ListModelView[0]
    self.CurrentModelName = model.name
    self.CurrentModelPrototypes = str(model.prototype_source)
    self.CurrentModelLayer = str(model.layer)
    
  def _AddModelButton_fired(self):
    model_class = models.GetModelClass()
    params = model_class.ParamClass()
    params.configure_traits(kind = 'modal')
    TempModel = ExpModel(params = params)
    TempModel.configure_traits(kind = 'modal')
    self.ListModel.append((TempModel,TempModel.name))
  
  def _AddCorpusButton_fired(self):
    temp = TempDir()
    temp.configure_traits(kind = 'modal')
    try:
      train, test = SetCorpus(temp.value)
      self.ListCorpus = self.ListCorpus + [(train,train.name), (test, test.name)]  
    except:
      pass
  def _AddClassifierButton_fired(self):
    name = TempStr()
    name.configure_traits(kind = 'modal')
    classifier = ExpClassifier(name = str(name.value))
    self.ListClassifier = self.ListClassifier + [(classifier,classifier.name)]
    
  def _ImprintButton_fired(self):
    model = self.ListModelView[0]
    corpus = self.ListCorpusView[0]
    model.ImprintS2Prototypes(corpus, 100)
    self._ListModelView_changed()
    
  def _MakePrototypesButton_fired(self):
    model = self.ListModelView[0]
    corpus = self.ListCorpusView[0]
    funcStr = self.ListPrototypeSources[0]
    func = getattr (model, funcStr)
    print numberofPrototypes
    func(corpus, 100)
    self._ListModelView_changed()
  def _ComputeFeatureButton_fired(self):
    model = self.ListModelView[0]
    corpus = self.ListCorpusView[0]
    model.ComputeFeatures(corpus)
    
  def _TrainSvmButton_fired (self):
    corpus = self.ListCorpusView[0]
    classifier = self.ListClassifierView[0]
    classifier.TrainSvm(corpus)
  
  def _TestSvmButton_fired (self, info):
    corpus = self.ListCorpusView[0]
    classifier = self.ListClassifierView[0]
    if corpus.features == None:
      self._ComputeFeatureButton_fired()
    self.Result = str(classifier.TestSvm(corpus))

class Controller(Handler):
  def Nothing(self, info):
    pass
Main = Group(Group(Item('ListModelView'), Item('AddModelButton', label = ' '), orientation = 'horizontal'), 
              Group(Item('ListCorpusView'), Item ('AddCorpusButton', label = ' '), orientation = 'horizontal'), 
              Group(Item('ListClassifierView'), Item('AddClassifierButton', label = ' '), orientation = 'horizontal'), Item('Result', style = 'readonly'))

NewMain = Group(Item('ListModelView'), Item('ListCorpusView'), Item('ListClassifierView'))
Utility = Group (Item('ImprintButton'), Item('ComputeFeatureButton'), Item('TrainSvmButton'), Item('TestSvmButton'))
Model = Group(Item('AddModelButton'), 
              Group(Item('ListPrototypeSources'), Item('MakePrototypesButton', label = ' '), orientation = 'horizontal'),
              Item('CurrentModelName'), 
              Item('CurrentModelPrototypes', style = 'readonly'), 
              Item('CurrentModelLayer'),
              label = 'Model')
Corpus = Group(Item('AddCorpusButton'), label = 'Corpus')
Classifier = Group(Item('AddClassifierButton'), Item('TrainSvmButton'), Item('TestSvmButton'), label = 'Classifier')

NewUtility = Group (Model, Corpus, Classifier, layout = 'tabbed')
view = View(Group(NewMain, Item('_'),
                  NewUtility,
                  orientation = 'horizontal'),
            title = 'Glimpse',
            handler = Controller)

Exp = UI()
Exp.configure_traits(view = view)
