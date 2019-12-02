from RunManager import RunManager
from collections import OrderedDict, namedtuple
from itertools import product


# class RunBuilder():
#     @staticmethod
#     def get_runs(params):

#         Run = namedtuple('Run', params.keys())

#         runs = []
#         for v in product(*params.values()):
#             runs.append(Run(*v))

#         return runs


params = OrderedDict(
    ls=[.01],
    batch_size=[1000, 2000],
    shuffle=[True, False]
)

m = RunManager()
for run in RunManager.get_runs(params):

    network = Network()
    loader = Dataloader(
        train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:

            images = batch[0]
            labels = batch[1]
            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
m.save('results')
