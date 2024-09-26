import jax
import jax.numpy as jnp
from functools import partial

class ReintegrationTracking:
    def __init__(self, SX=256, SY=256, dt=0.2, dd=5, sigma=0.65, border="wall", has_hidden=False, mix="stoch"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.border = border
        self.has_hidden = has_hidden
        self.mix = mix

        # Pre-compute static values
        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        self.pos = jnp.dstack((Y, X)) + 0.5  # (SX, SY, 2)

        dxs, dys = [], []
        for dx in range(-self.dd, self.dd+1):
            for dy in range(-self.dd, self.dd+1):
                dxs.append(dx)
                dys.append(dy)
        self.dxs, self.dys = jnp.array(dxs), jnp.array(dys)

        self.ma = self.dd - self.sigma  # upper bound of the flow magnitude

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, A: jax.Array, F: jax.Array) -> jax.Array:
        return self._apply_without_hidden(A, F)
    
    @partial(jax.jit, static_argnums=(0,))
    def _apply_without_hidden(self, A: jax.Array, F: jax.Array) -> jax.Array:
        mu = self.pos[..., None] + jnp.clip(self.dt * F, -self.ma, self.ma)
        mu = jax.lax.cond(
            self.border == "wall",
            lambda x: jnp.clip(x, self.sigma, self.SX-self.sigma),
            lambda x: x,
            mu
        )
        nA = jax.vmap(self._step, in_axes=(None, None, 0, 0))(A, mu, self.dxs, self.dys).sum(0)
        return nA
    
    @partial(jax.jit, static_argnums=(0,))
    def _apply_without_hidden(self, A: jax.Array, F: jax.Array) -> jax.Array:
        mu = self.pos[..., None] + jnp.clip(self.dt * F, -self.ma, self.ma)  # (x, y, 2, c): target positions
        
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)

        nA = self._step(A, mu, self.dxs, self.dys).sum(0)
        return nA

    @partial(jax.jit, static_argnums=(0,))
    def _apply_with_hidden(self, A: jax.Array, H: jax.Array, F: jax.Array):
        mu = self.pos[..., None] + jnp.clip(self.dt * F, -self.ma, self.ma)  # (x, y, 2, c): target positions
        
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)

        nA, nH = self._step_flow(A, H, mu, self.dxs, self.dys)

        if self.mix == 'avg':
            nH = jnp.sum(nH * nA.sum(axis=-1, keepdims=True), axis=0)  
            nA = jnp.sum(nA, axis=0)
            nH = nH / (nA.sum(axis=-1, keepdims=True) + 1e-10)
        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis=-1, keepdims=True)) - 1
            nA = jnp.sum(nA, axis=0)
            nH = jnp.sum(nH * expnA, axis=0) / (expnA.sum(axis=0) + 1e-10)
        elif self.mix == "stoch":
            categorical = jax.random.categorical(
                jax.random.PRNGKey(42), 
                jnp.log(nA.sum(axis=-1, keepdims=True)), 
                axis=0)
            mask = jax.nn.one_hot(categorical, num_classes=(2*self.dd+1)**2, axis=-1)
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nH = jnp.sum(nH * mask, axis=0)
            nA = jnp.sum(nA, axis=0)

        return nA, nH

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, A, mu, dxs, dys):
        def roll_and_compute(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            
            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(self.pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis=0)
            else:
                dpmu = jnp.absolute(self.pos[..., None] - mur)
            
            sz = 0.5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)), axis=2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA

        return jax.vmap(roll_and_compute, in_axes=(None, None, 0, 0))(A, mu, dxs, dys)

    @partial(jax.jit, static_argnums=(0,))
    def _step_flow(self, A, H, mu, dx, dy):
        Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
        Hr = jnp.roll(H, (dx, dy), axis=(0, 1))  # (x, y, k)
        mur = jnp.roll(mu, (dx, dy), axis=(0, 1))

        if self.border == 'torus':
            dpmu = jnp.min(jnp.stack(
                [jnp.absolute(self.pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
            ), axis=0)
        else:
            dpmu = jnp.absolute(self.pos[..., None] - mur)

        sz = 0.5 - dpmu + self.sigma
        area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)), axis=2) / (4 * self.sigma**2)
        nA = Ar * area
        return nA, Hr