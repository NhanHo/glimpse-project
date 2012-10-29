from traits.api import  Str, Int, List, Button, HasTraits
from traitsui.api import View, Item, CheckListEditor, Group, Handler, Action
from .experiment import ExpModel, ExpCorpus, ExpClassifier
from .misc import SetCorpus
class ExperimentUI(HasTraits):
  ListModel = List([])
  ListModelName = List([])
  ListCorpus = [1,2,3]
  ListCorpusName = ['a','b','c']
  ListClassifier = [ExpClassifier()]
  ListClassifierName = ['ClassifierSVM']
  TestListModelView = List(['a','b'])
  ListModelView = List (editor = CheckListEditor(name = 'ListModelName')) 

  def __init__(self):
    self.ListModel += [ExpModel()]
    self.ListCorpusView = List (editor = CheckListEditor(values = zip(self.ListCorpus,self.ListCorpusName)))
    self.ListClassifierView = List (editor = CheckListEditor(values = zip(self.ListClassifier,self.ListClassifierName)))
    print self.ListModelName
	
    
  def _ListModel_changed(self):
    print "aaa"
    self.ListModelName = [(x,x.name) for x in self.ListModel]
    
  CurrentModel = None
  CurrentCorpus = None
  CurrentClassifier = None
  
  AddModel = Button('ABCD')
  EditModel = Button ('EFGH')
  ImprintModelButton = Button ('Imprint S2 Prototypes')
  ComputeFeatureButton = Button ('Compute image features')
  SetCorpusButton = Button ('Set Corpus')
  
  Result = Str(1)
  cl_4_group = Item( 'ListModelView')
  
  def _ListCorpus_changed(self):
    print "New Corpus added"
    
  def _AddModel_fired(self):
    #self.CurrentModel[0].configure_traits()
    self.ListModel = self.ListModel +  [ExpModel()]

  def _SetCorpusButton_fired(self):
    print "Adding new corpus"
    train, test = SetCorpus('corpus')
    self.AddCorpus(train)
    self.AddCorpus(test)
    
  def AddCorpus (self, newcorpus):
    self.ListCorpus = self.ListCorpus + [newcorpus]
    self.ListCorpusName = self.ListCorpusName + [newcorpus.name]
    
class Controller(Handler):
  def Classify1(self, info):
    Classifier =  info.object.ListClassifierView
    Corpus = info.object.ListCorpusView
    info.object.Result = Str (Classifier.TestSvm(Corpus))
      
  def object_ListModelView_changed(self, info):
    #info.object.ListModelView = List (editor = CheckListEditor(values = zip(info.object.ListModel,info.object.ListModelName)))
    info.object.CurrentModel = info.object.ListModelView
    print info.object.CurrentModel

  def object_ListCorpusView_changed (self, info):
    #info.ListCorpusView.factory.values =zip(info.object.ListCorpus,info.object.ListCorpusName)
    info.object.CurrentCorpus = info.object.ListCorpusView
    
  def object_ListClassifierView_changed (self, info):
    #info.object.ListClassifierView = List (editor = CheckListEditor(values = zip(info.object.ListClassifier,info.object.ListClassifierName)))

    info.object.CurrentClassifier = info.object.ListClassifierView
    
  def AddModel(self, info):
    info.object.ListModel[0].configure_traits()

    
GroupModelView = Group(Item('ListModelView'), \
				Item(name = 'AddModel', label = ' '), Item(name = 'EditModel', label = ' '), columns = 2)
GroupMainAction = Group(GroupModelView, Item ('ListCorpusView'), Item ('ListClassifierView'))
GroupUtility = Group (Item('AddModel') , Item('ImprintModelButton', label = 'Imrpint S2 Prototypes'), Item ('ComputeFeatureButton'), Item('SetCorpusButton'))
Exp = ExperimentUI()


Classify1 = Action( name = 'Classify', action = 'Classify1')
view1 = View(Group(GroupMainAction, Item('_'), GroupUtility, orientation = 'horizontal'),
          title = 'Test View',
          buttons = ['OK', Classify1],
          resizable = True,
          handler = Controller())


