// The narration, scene by scene. The picture and the music are timed from it:
// voice.mjs speaks each line and measures it; timeline.js lays the lines end to end.
//   lead: seconds of picture before the first line; tail: after the last one
//   a line's gap: the pause after it (default 0.45 s)
//   say: what the voice reads when it differs from what is written (pronunciation, numbers)
(function (root) {
  const SCRIPT = [
    { id: 'hook', lead: 0.8, tail: 0.9, lines: [
      { id: 'h1', text: 'Seventeen million parameters. Forty-two seconds of training.', gap: 0.5 },
      { id: 'h2', text: 'And it sorts bank messages about as well as Claude Opus.', gap: 0.9 },
      { id: 'h3', text: 'This chart got a lot of attention, and many of you asked the same things. What is this model? Where does the data come from? And when is it worth the trouble?', gap: 0.6 },
      { id: 'h4', text: 'Here is the long version: what we measured, and how to do it yourself.' },
    ] },
    { id: 'job', lead: 0.7, tail: 0.8, lines: [
      { id: 'j1', text: 'First, the job.', gap: 0.4 },
      { id: 'j2', text: 'A bank gets thousands of messages a day, and each one must land in one of seventy-seven boxes: card arrival, lost card, refund not showing.', gap: 0.5 },
      { id: 'j3', text: 'That’s a classifier. They are everywhere: support tickets, emails, contracts, reviews.', gap: 0.5 },
      { id: 'j4', text: 'Today, there are three ways to get one.' },
    ] },
    { id: 'ways', lead: 0.5, tail: 1.0, lines: [
      { id: 'w1', text: 'One: ask a frontier AI, like Claude Opus or Kimi. Very smart, but slow, and every answer costs money.', say: 'One: ask a frontier AI, like Claude Opus or Kimi. Very smart, but slow, and every answer costs money.', gap: 0.5 },
      { id: 'w2', text: 'Two: a classification API, like Jev. You give it your categories, and it answers in a tenth of a second. No training at all.', say: 'Two: a classification A P I, like [Jev](/ʤˈɛv/). You give it your categories, and it answers in a tenth of a second. No training at all.', gap: 0.5 },
      { id: 'w3', text: 'Three: train your own tiny model. That used to mean days of work. So we measured what it takes now.' },
    ] },
    { id: 'result', lead: 0.6, tail: 1.2, lines: [
      { id: 'r1', text: 'Here is banking77: ten thousand real bank messages, each one labelled by a person.', say: 'Here is banking seventy-seven: ten thousand real bank messages, each one labelled by a person.', gap: 0.5 },
      { id: 'r2', text: 'We fine-tuned Ettin, a model with seventeen million parameters, on those labels. On one gaming GPU from 2020, it took forty-two seconds.', say: 'We fine-tuned [Ettin](/ˈɛtɪn/), a model with seventeen million parameters, on those labels. On one gaming G P U from twenty twenty, it took forty-two seconds.', gap: 0.6 },
      { id: 'r3', text: 'On three thousand messages it had never seen, it got 91.5% right. Opus got 92. Kimi, 82. And Jev, 79.', say: 'On three thousand messages it had never seen, it got ninety-one and a half percent right. Opus got ninety-two. Kimi, eighty-two. And [Jev](/ʤˈɛv/), seventy-nine.', gap: 0.6 },
      { id: 'r4', text: 'To sort a million messages, Opus would cost about eight thousand dollars. The tiny model costs a few cents of electricity, and answers in five milliseconds.' },
    ] },
    { id: 'ettin', lead: 0.6, tail: 1.1, lines: [
      { id: 'e1', text: 'So what is Ettin? An encoder, a cousin of BERT, released openly by researchers at Johns Hopkins.', say: 'So what is [Ettin](/ˈɛtɪn/)? An encoder, a cousin of BERT, released openly by researchers at Johns Hopkins.', gap: 0.5 },
      { id: 'e2', text: 'It doesn’t write text. It reads the whole message at once, and gives one score per category.', gap: 0.5 },
      { id: 'e3', text: 'To fine-tune it, you add one output per category, and train every weight on your examples. Six passes over the data. That’s the whole trick.' },
    ] },
    { id: 'anywhere', lead: 0.6, tail: 1.2, lines: [
      { id: 'a1', text: 'And you don’t even need a GPU.', say: 'And you don’t even need a G P U.', gap: 0.5 },
      { id: 'a2', text: 'The same recipe took eleven minutes on a 2019 laptop, and twenty-one minutes on a phone’s graphics chip, inside Chrome.', say: 'The same recipe took eleven minutes on a twenty nineteen laptop, and twenty-one minutes on a phone’s graphics chip, inside Chrome.', gap: 0.5 },
      { id: 'a3', text: 'Same accuracy, within half a point. At this size, there’s nothing to host: the model lives inside your app.' },
    ] },
    { id: 'curve', lead: 0.6, tail: 1.2, lines: [
      { id: 'd1', text: 'But here is the catch. The model is the easy part. The labels are the whole game.', gap: 0.6 },
      { id: 'd2', text: 'Watch what happens with fewer of them.', gap: 0.4 },
      { id: 'd3', text: 'A hundred examples: fifteen percent right. A thousand: sixty-seven. Two thousand: eighty-one. Four thousand: eighty-seven.', gap: 0.6 },
      { id: 'd4', text: 'And no curve has flattened yet. More labels would still help.' },
    ] },
    { id: 'distill', lead: 0.6, tail: 1.1, lines: [
      { id: 't1', text: 'Most people don’t have ten thousand labelled messages. So, borrow a teacher.', gap: 0.5 },
      { id: 't2', text: 'Ask a big model, here Kimi K3, to label your messages. Then train the tiny model on its answers. That’s distillation.', say: 'Ask a big model, here Kimi K three, to label your messages. Then train the tiny model on its answers. That’s distillation.', gap: 0.5 },
      { id: 't3', text: 'Spread over four providers at once, Kimi labelled all nine and a half thousand messages in twenty-eight seconds, for fourteen dollars.', gap: 0.5 },
      { id: 't4', text: 'Just check the teacher’s terms first. Kimi’s allow training on its answers. Opus’s and Jev’s don’t.', say: 'Just check the teacher’s terms first. Kimi’s allow training on its answers. Opus’s and [Jev’s](/ʤˈɛvz/) don’t.' },
    ] },
    { id: 'ceiling', lead: 0.6, tail: 1.3, lines: [
      { id: 'c1', text: 'So, how good is the student?', gap: 0.5 },
      { id: 'c2', text: 'With a few hundred examples, Kimi’s labels are as good as human ones. Then the lines split.', gap: 0.5 },
      { id: 'c3', text: 'With all the data, the model taught by people reaches 91.5. The one taught by Kimi stops at 79.5, just under Kimi itself.', say: 'With all the data, the model taught by people reaches ninety-one and a half. The one taught by Kimi stops at seventy-nine and a half, just under Kimi itself.', gap: 0.6 },
      { id: 'c4', text: 'This held on every task. A student lands a few points under its teacher: it learns the teacher’s mistakes too. Human labels break that ceiling.' },
    ] },
    { id: 'soft', lead: 0.6, tail: 1.2, lines: [
      { id: 's1', text: 'Some of you asked: what if the student copies the teacher’s probabilities, not just its answer?', gap: 0.5 },
      { id: 's2', text: 'We tested it. With plenty of data, the accuracy doesn’t move.', gap: 0.5 },
      { id: 's3', text: 'But its confidence becomes honest. Trained on answers only, it claims ninety-five percent certainty, while being right seventy-seven percent of the time.', gap: 0.5 },
      { id: 's4', text: 'Trained on probabilities, its most confident half is right ninety-six percent of the time. Keep those, and send the rest to a bigger model.' },
    ] },
    { id: 'general', lead: 0.6, tail: 1.4, lines: [
      { id: 'g1', text: 'Was banking77 a lucky case? We ran the same recipe, untouched, on ten different tasks.', say: 'Was banking seventy-seven a lucky case? We ran the same recipe, untouched, on ten different tasks.', gap: 0.5 },
      { id: 'g2', text: 'On human labels, the tiny model beat Jev on six of them.', say: 'On human labels, the tiny model beat [Jev](/ʤˈɛv/) on six of them.', gap: 0.5 },
      { id: 'g3', text: 'On Kimi’s labels, it tied Jev on two, and stayed within five points on five more.', say: 'On Kimi’s labels, it tied [Jev](/ʤˈɛv/) on two, and stayed within five points on five more.', gap: 0.5 },
      { id: 'g4', text: 'It lost where Kimi itself was weaker than Jev, and on hate speech, where the test messages didn’t look like the training ones. Your examples must look like your real data.', say: 'It lost where Kimi itself was weaker than [Jev](/ʤˈɛv/), and on hate speech, where the test messages didn’t look like the training ones. Your examples must look like your real data.' },
    ] },
    { id: 'pays', lead: 0.6, tail: 1.3, lines: [
      { id: 'p1', text: 'So when is it worth it? Let’s count.', gap: 0.5 },
      { id: 'p2', text: 'Up front, the labels cost between one and thirty dollars, and training takes one to three minutes. After that, each message is almost free.', gap: 0.5 },
      { id: 'p3', text: 'Against Jev, that pays back somewhere between fifty thousand and half a million messages.', say: 'Against [Jev](/ʤˈɛv/), that pays back somewhere between fifty thousand and half a million messages.', gap: 0.5 },
      { id: 'p4', text: 'Below that, use Jev. Above that, or when you need speed, privacy, or no internet: fine-tune.', say: 'Below that, use [Jev](/ʤˈɛv/). Above that, or when you need speed, privacy, or no internet: fine-tune.' },
    ] },
    { id: 'twist', lead: 0.6, tail: 1.3, lines: [
      { id: 'x1', text: 'One reply to the post says it best.', gap: 0.5 },
      { id: 'x2', text: 'To trust any model in production, Jev included, you need examples with known answers, to check it.', say: 'To trust any model in production, [Jev](/ʤˈɛv/) included, you need examples with known answers, to check it.', gap: 0.5 },
      { id: 'x3', text: 'And once you have those, you are most of the way to training your own.', gap: 0.7 },
      { id: 'x4', text: 'Jev is like Python, and a fine-tuned model is like Rust. Start in Python. Compile what proves it needs to scale.', say: '[Jev](/ʤˈɛv/) is like Python, and a fine-tuned model is like Rust. Start in Python. Compile what proves it needs to scale.' },
    ] },
    { id: 'recipe', lead: 0.6, tail: 0.6, lines: [
      { id: 'k1', text: 'Here is the recipe.', gap: 0.5 },
      { id: 'k2', text: 'Write your categories. Take two thousand real messages. Have people, or an allowed teacher, label them.', gap: 0.4 },
      { id: 'k3', text: 'Keep a few hundred aside, checked by a person. That’s your test.', gap: 0.4 },
      { id: 'k4', text: 'Fine-tune Ettin: a minute on a GPU, ten on a laptop. Compare it with Jev and a frontier model on your test, and keep the winner.', say: 'Fine-tune [Ettin](/ˈɛtɪn/): a minute on a G P U, ten on a laptop. Compare it with [Jev](/ʤˈɛv/) and a frontier model on your test, and keep the winner.', gap: 0.5 },
      { id: 'k5', text: 'When your messages change, label a few more and train again. It takes a minute.' },
    ] },
    { id: 'close', lead: 0.5, tail: 3.2, lines: [
      { id: 'z1', text: 'Evaluate first. Fine-tune what pays.' },
    ] },
  ];
  root.SCRIPT = SCRIPT;
})(typeof window !== 'undefined' ? window : globalThis);
