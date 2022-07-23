
1. Compute spectrogram with short-time Fourier transform
	[Spectrogram Computation in Signal Analyzer](https://www.mathworks.com/help/signal/ug/spectrogram-computation-in-signal-analyzer.html)
	
	input: waveform $x$
	output: spectrogram $y∈[0,1]^{h×t×c}$
	
	$y∈[0,1]^{h×t×c}$ is spectrogram image. 
	$h$ is frequency. $t$ is time. $c$ is channel (should be 1)

2. Compute formant of each time with encoder
	$$freq_{1:M},strength_{1:M},bandwith_{1:M}=h^{enc}(y^{t})$$
	$h^{enc}$ is a $n×1$ convolution encoder?
	output: frequency of formants, amplitude of formants, and the formant bandwith

3. 

	1. Reconstruct formant into spectrogram with formant synthesizer
		[KlattSyn - Klatt Formant Synthesizer](https://github.com/chdh/klatt-syn)
		
		![[Klatt Formant Synthesizer.png]]
		
		$loss1=|y'-y^{t}|$
		$y'$ is the reconstructed spectrogram
		
	2. Predict text with transformer decoder
		
		$p^{t}=Transformer(freq_{1:M}^{t},strength_{1:M}^{t},bandwith_{1:M}^{t})$

4. 

	$While\ \{a_{t}^{l}\ != a_{t+1}^{l}\ or\ a_{t}^{l+1} != a_{t+1}^{l+1}\}$
	
	$C_{t+1}^{l+1} = g(a_{t}^{l+1}) = 
	sum(\begin{bmatrix}
	C_{t}^{l+1} W_{v}^{l+1}
	\\
	repeat(expand(C_{t}^{l} W_{v}^{l}),m)
	\end{bmatrix} \odot Soft-max(a_{t}^{l+1},dim=0),dim=0)$
	
	
	$C_{t+1}^{l} = g(a_{t}^{l}) = 
	sum(\begin{bmatrix}
	C_{t}^{l} W_{v}^{l}
	\\
	repeat(expand(C_{t}^{l+1} W_{v}^{l+1}),n)
	\end{bmatrix} \odot Soft-max(a_{t}^{l},dim=0),dim=0)$
	
	
	
	
	
	$a_{t+1}^{l+1} = f(C_{t+1}^{l+1}) = f(g(a_{t}^{l+1})) = 
	\begin{bmatrix}
	(g(a_{t}^{l+1})  W_{q}^{l+1} ) \otimes (g(a_{t}^{l+1})  W_{k}^{l+1} )
	\\
	(g(a_{t}^{l+1})  W_{q}^{l+1} ) \otimes (C_{t}^{l}  W_{k}^{l} )
	\end{bmatrix}$
	
	
	$a_{t+1}^{l} = f(C_{t+1}^{l}) = f(g(a_{t}^{l})) = 
	\begin{bmatrix}
	(g(a_{t}^{l})  W_{q}^{l} ) \otimes (g(a_{t}^{l})  W_{k}^{l} )
	\\
	(g(a_{t}^{l})  W_{q}^{l} ) \otimes (g(a_{t}^{l})  W_{k}^{l+1} )
	\end{bmatrix}$
	
	$EndWhile$
	
	
	
	$a^{l} \in (n, m), l=(0,1,2,...,L-1)$
	
	$a_{t+1}^{l} = [W_{q}^{l+1} \cdot Capsule_{t+1}^{l+1}] \cdot [W_{q}^{l} \cdot Capsule_{t+1}^{l}]$
	
	Input: 
	$L$ capsule layers
	$a$: botton-up relation
	$a'$: top-down relation
	
	$v = W_{v} \cdot Capsule_{t}$
	
	$Capsule_{t+1} = Capsule_{t} + v^{[0,L)} \cdot a_{t} + v^{(0,L]} \cdot a'_{t}$
	
	$q = W_{q} \cdot Capsule_{t+1}$
	
	$k = W_{k} \cdot Capsule_{t+1}$
	
	$a = q^{[0,L)} \cdot k^{(0,L]}$, $a' = q^{(0,L]} \cdot k^{[0,L)}$
	
	
	
	$Capsule_{t+1}^{l+1} = W_{v}^{l+1} \cdot softmax(a_{t}^{l})$
	
	$a_{t+1}^{l} = [W_{q}^{l+1} \cdot W_{v}^{l+1} \cdot softmax(a_{t}^{l})] \cdot [W_{q}^{l} \cdot W_{v}^{l} \cdot softmax(a_{t}^{l})]$
	
	
	
	$a_{t+1}^{l+1} = f(Capsule_{t+1}^{l+1},Capsule_{t+1}^{l})$ 
	
	$a_{t+1}^{l} = f(Capsule_{t+1}^{l+1},Capsule_{t+1}^{l})$
	
	$iter=t,t+1$
	
	$layer=l,l+1$
	

5. 失败设计（太混乱复杂了，但其实是对的）
	
	\begin{algorithm}
	\caption{Attention routing between capsule layers}\label{alg:cap}
	\begin{algorithmic}
	\Require 
	$C_{0:n}^{l}\in\mathbb{R}^{(v)}$, $C_{0:m}^{l+1}\in\mathbb{R}^{(v')}$,  \Comment{Output capsules of layer $l$ and $l+1$}
	
	$i\in(0,1,...,n),v^{l}=(Prob, hiddn\_feats^{a})$
	
	$j\in(0,1,...,m),v^{l+1}=(Prob, hiddn\_feats^{b})$
	
	$W_{q}^{l}, W_{k}^{l}, W_{v}^{l}$  \Comment{attention matrix}
	
	$W_{q}^{l+1}, W_{k}^{l+1}, W_{v}^{l+1}$
	
	\State $iter \gets 0$
	\While{$iter < max\_iteration$}
	\State //\ Calculate $q,k,v$
	\State $q^{l},k^{l},v^{l}=W_{q}^{l}C^{l}[1:],W_{k}^{l}C^{l}[1:],W_{v}^{l}C^{l}[1:]$
	\State $q^{l+1},k^{l+1},v^{l+1}=W_{q}^{l+1}C^{l+1}[1:],W_{k}^{l+1}C^{+1l}[1:],W_{v}^{l+1}C^{l+1}[1:]$
	\State $a^{l+1|l} = Softmax(q^{l} \cdot k^{l+1})$  \Comment{the relation between two layers from lower perspective}
	\State $a^{l|l+1} = Softmax(q^{l+1} \cdot k^{l})$  \Comment{the relation between two layers from higher perspective}
	\State //\ Update higher-layer capsules $C_{0:m}^{l+1}$
	\State $index \gets 0$
	\While{$index < m$}
	\State $C_{index}^{l+1}[1:] = q_{index}^{l+1} \cdot k_{index}^{l+1} + a^{l+1|l} \cdot v^{l}$  \Comment{get features from itself and lower capsules}
	\EndWhile
	\State //\ Update lower-layer capsules $C_{0:m}^{l}$
	\State $index \gets 0$
	\While{$index < n$}
	\State $C_{index}^{l}[1:] = q_{index}^{l} \cdot k_{index}^{l} + a^{l|l+1} \cdot v^{l+1}$
	\EndWhile
	\EndWhile
	\State //\ Settle probability $C_{0:m}^{l+1}[0]$ after iterations
	\State $C_{0:m}^{l+1}[0] = a^{l|l+1}$
	
	\State //\ Calculate Cross-entropy Loss
	\State $loss = MSE(a^{l+1|l}, a^{l|l+1})$
	\end{algorithmic}
	\end{algorithm}

