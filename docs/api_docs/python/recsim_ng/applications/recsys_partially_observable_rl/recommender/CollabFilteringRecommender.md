description: A collaborative filtering based recommender implementation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.applications.recsys_partially_observable_rl.recommender.CollabFilteringRecommender" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="slate_docs"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.applications.recsys_partially_observable_rl.recommender.CollabFilteringRecommender

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/recommender.py">View
source</a>

A collaborative filtering based recommender implementation.

Inherits From:
[`BaseRecommender`](../../../../recsim_ng/entities/recommendation/recommender/BaseRecommender.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.applications.recsys_partially_observable_rl.recommender.CollabFilteringRecommender(
    config: <a href="../../../../recsim_ng/core/value/Config.md"><code>recsim_ng.core.value.Config</code></a>,
    model_ctor: tf.keras.Model = <a href="../../../../recsim_ng/applications/recsys_partially_observable_rl/recommender/CollabFilteringModel.md"><code>recsim_ng.applications.recsys_partially_observable_rl.recommender.CollabFilteringModel</code></a>,
    name=&#x27;Recommender&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
a descriptive name identifying the entity.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `name` </td> <td> Returns the name of this module as passed or
determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes parent
module names. </td> </tr><tr> <td> `name_scope` </td> <td> Returns a
`tf.name_scope` instance for this class. </td> </tr><tr> <td>
`non_trainable_variables` </td> <td> Sequence of non-trainable variables owned
by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`submodules` </td> <td> Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```

</td> </tr><tr> <td> `trainable_variables` </td> <td> Sequence of trainable
variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`variables` </td> <td> Sequence of variables owned by this module and its
submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>

## Methods

<h3 id="initial_state"><code>initial_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/recommender.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The initial state value.

<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/recommender.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    previous_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    user_response: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    slate_docs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The state value after the initial value.

<h3 id="slate_docs"><code>slate_docs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/recommender.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>slate_docs(
    previous_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    user_obs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    available_docs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The slate_docs value.

<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/recommender.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>specs() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose names
included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>
