\# Task C — Failure, Debugging, and Iteration



\## Problem



During live testing of SignBridge, holding a single hand gesture caused the same

letter to be inserted repeatedly into the word builder.



For example, holding one gesture could produce sequences such as:



B B B B D D D N N S S



The underlying recognition model was producing predictions continuously for

successive webcam frames, while the auto-add logic treated those predictions as

independent character events.



\## Initial Fix



I changed the auto-add decision so that the same stable prediction could not be

inserted repeatedly while the gesture was being held.



The system now tracks the last automatically inserted prediction and requires a

different stable prediction before inserting another character.



\## Second Issue Discovered



During further testing, the first fix exposed a deeper temporal problem:

the model could occasionally change its prediction between visually similar

letters while the user was holding the same gesture.



This caused sequences such as:



A → E → A → H



to potentially enter the word builder.



\## Revised Solution



I improved the temporal decision layer in `src/smoothing.py`.



The updated `TemporalSmoother` combines:



1\. Consecutive prediction confirmation.

2\. Majority voting over recent predictions.

3\. State reset when the hand disappears.



A prediction can therefore become stable through repeated consecutive evidence

or majority evidence rather than relying on a single frame.



\## Validation



The existing automated test suite was executed after the change:



&#x20;   4 passed, 1 warning



The warning came from an existing Keras API recommendation and did not cause

test failure.



\## Engineering Learning



The main lesson was that frame-level model accuracy does not automatically

translate into stable real-time interaction.



A real-time gesture application needs a decision layer between model inference

and user-visible actions. Temporal reasoning and event-based input handling

are necessary to convert noisy frame predictions into reliable user actions.

