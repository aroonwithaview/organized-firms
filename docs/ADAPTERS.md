# Connecting a deployed mediator

The AI mediator institution takes a `PlatformMediator` (see `orgsim/interfaces.py`). To evaluate a deployed system that coaches suppliers, implement:

```python
from orgsim.interfaces import PlatformMediator

class DeployedMediator(PlatformMediator):
    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num,
                          buyer_description="", previous_bids=None, previous_feedback=None) -> str:
        # Send the supplier's current bid to the deployed system and return its advice as text.
        raise NotImplementedError
```

Then pass an instance of it to `run_sequence(..., mediator=DeployedMediator())`.

What each call receives: the buyer's preferences; every supplier's current bid (`all_bids`); every supplier's bids so far in this event, the current round's included (`previous_bids`); your feedback to every supplier in earlier rounds of this event (`previous_feedback`, which never holds a reply from the current round; within a round, every call gets the same copy, taken before the round's first call); and, in `buyer_description`, the buyer's scoring function preceded by the platform's memory of every supplier's past events. Return the feedback for `target_supplier_id` as non-empty text. If the deployed system gives no usable reply, raise an exception: the package's own mediator does, so that a failure stops the run instead of reaching the supplier as silence. The rules suppliers read under the advice prompts say the platform does not disclose how many suppliers take part, so the adapter should not pass on a count either. Under `--prompts primitives` the suppliers are told that one other supplier takes part and that the platform sees every supplier's bids in the event, the buyer's scoring and past events; they are not told the other supplier's bids or costs, so the adapter must not pass on any.

Supplier costs are assigned by the researcher, so every outcome reached with the deployed system can be scored against the efficient frontier.

Keep credentials and vendor-specific API details out of this repository.

This environment has not been run against any deployed system. All recorded runs use the repository's own AI mediator and AI buyer's agent, and no adapter was used. The `PlatformMediator` interface is there so that a researcher can connect a deployed intermediary; the class above is a template and has not been exercised against one.
