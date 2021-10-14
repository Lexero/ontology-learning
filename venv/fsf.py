from ontolearn import KnowledgeBase,SampleConceptLearner
from ontolearn.metrics import F1, PredictiveAccuracy, CELOEHeuristic,DLFOILHeuristic
kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')

p = {'http://www.benchmark.org/family#F10M173', 'http://www.benchmark.org/family#F10M183'}
n = {'http://www.benchmark.org/family#F1F5', 'http://www.benchmark.org/family#F1F7'}

model = SampleConceptLearner(knowledge_base=kb,
                             quality_func=F1(),
                             terminate_on_goal=True,
                             heuristic_func=DLFOILHeuristic(),
                             iter_bound=100,
                             verbose=False)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)

######################################################################
model = SampleConceptLearner(knowledge_base=kb,
                             quality_func=PredictiveAccuracy(),
                             terminate_on_goal=True,
                             heuristic_func=CELOEHeuristic(),
                             iter_bound=100,
                             verbose=False)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)