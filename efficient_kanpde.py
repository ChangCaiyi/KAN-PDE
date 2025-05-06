import torch
import torch.nn.functional as F
import math


def compute_dudx(u, dx, is_periodic=True):
    # u.shape = (Nx,)
    Nx = u.shape[-1]
    if Nx<=5:
        raise ValueError('Must have at least 6 spatial points.')

    if is_periodic:
        ux = torch.zeros_like(u)
        ux[:,2:Nx-2] = (u[:,0:Nx-4] - 8*u[:,1:Nx-3] + 8*u[:,3:Nx-1] - u[:,4:Nx])/12/dx
        ux[:,0] = (u[:,-2] - 8*u[:,-1] + 8*u[:,1] - u[:,2])/12/dx
        ux[:,1] = (u[:,-1] - 8*u[:,0] + 8*u[:,2] - u[:,3])/12/dx
        ux[:,-2] = (u[:,-4] - 8*u[:,-3] + 8*u[:,-1] - u[:,0])/12/dx
        ux[:,-1] = (u[:,-3] - 8*u[:,-2] + 8*u[:,0] - u[:,1])/12/dx
    else:
        ux = torch.zeros_like(u)
        ux[:,2:Nx-2] = (u[:,0:Nx-4] - 8*u[:,1:Nx-3] + 8*u[:,3:Nx-1] - u[:,4:Nx])/12/dx
        for i in [0, 1]:
            ux[:,i] = (-25*u[:,i] + 48*u[:,i+1] - 36*u[:,i+2] + 16*u[:,i+3] - 3*u[:,i+4])/12/dx
        for i in [Nx-2, Nx-1]:
            ux[:,i] = -(-25*u[:,i] + 48*u[:,i-1] - 36*u[:,i-2] + 16*u[:,i-3] - 3*u[:,i-4])/12/dx
    return ux


def compute_dudx_2d(u, dx, is_periodic=True):
    Nx = u.shape[-2]
    if Nx<=5:
        raise ValueError('Must have at least 6 spatial points in x.')
    ux = torch.zeros_like(u)
    ux[...,2:Nx-2,:] = (u[...,0:Nx-4,:] - 8*u[...,1:Nx-3,:] + 8*u[...,3:Nx-1,:] - u[...,4:Nx,:])/12/dx

    if is_periodic:
        ux[...,-2,:] = (u[...,-4,:] - 8*u[...,-3,:] + 8*u[...,-1,:] - u[...,0,:])/12/dx
        ux[...,-1,:] = (u[...,-3,:] - 8*u[...,-2,:] + 8*u[...,0,:] - u[...,1,:])/12/dx
        ux[...,0,:] = (u[...,-2,:] - 8*u[...,-1,:] + 8*u[...,1,:] - u[...,2,:])/12/dx
        ux[...,1,:] = (u[...,-1,:] - 8*u[...,0,:] + 8*u[...,2,:] - u[...,3,:])/12/dx
    else:
        for i in [0, 1]:
            ux[...,i,:] = (-25*u[...,i,:] + 48*u[...,i+1,:] - 36*u[...,i+2,:] + 16*u[...,i+3,:] - 3*u[...,i+4,:])/12/dx
        for i in [-2, -1]:
            ux[...,i,:] = -(-25*u[...,i,:] + 48*u[...,i-1,:] - 36*u[...,i-2,:] + 16*u[...,i-3,:] - 3*u[...,i-4,:])/12/dx
    return ux


def compute_dudy_2d(u, dy, is_periodic=True):
    Ny = u.shape[-1]
    if Ny<=5:
        raise ValueError('Must have at least 6 spatial points in y.')
    uy = torch.zeros_like(u)
    uy[...,2:Ny-2] = (u[...,0:Ny-4] - 8*u[...,1:Ny-3] + 8*u[...,3:Ny-1] - u[...,4:Ny])/12/dy
    
    if is_periodic:
        uy[...,-2] = (u[...,-4] - 8*u[...,-3] + 8*u[...,-1] - u[...,0])/12/dy
        uy[...,-1] = (u[...,-3] - 8*u[...,-2] + 8*u[...,0] - u[...,1])/12/dy
        uy[...,0] = (u[...,-2] - 8*u[...,-1] + 8*u[...,1] - u[...,2])/12/dy
        uy[...,1] = (u[...,-1] - 8*u[...,0] + 8*u[...,2] - u[...,3])/12/dy

    else:
        for i in [0, 1]:
            uy[...,i] = (-25*u[...,i] + 48*u[...,i+1] - 36*u[...,i+2] + 16*u[...,i+3] - 3*u[...,i+4])/12/dy
        for i in [-2, -1]:
            uy[...,i] = -(-25*u[...,i] + 48*u[...,i-1] - 36*u[...,i-2] + 16*u[...,i-3] - 3*u[...,i-4])/12/dy
    return uy


# KAN-PDE is modified from https://github.com/Blealtan/efficient-kan.

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape                
        x = x.reshape(-1, self.in_features)     

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    
class KANPDE(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,  # number of neurons in each hidden layer
        dx,             # spatial grid size in x to compute the spatial derivatives  
        dy=None,        # spatial grid size in y to compute the spatial derivatives  
        pde_order=4,    # the order of the highest spatial derivative 
        pde_component=1,# number of components in a PDE system
        is_periodic=True, # whether periodic bc is applied for spatial derivatives 
        spatial_dim=1,  # spatial dimension
        mult_arity = 2, # number of multiplication arity for each multiplication node
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        '''
        KAN class
        
        layers_hidden : list
            number of neurons in each hidden layerß
            Without multiplication nodes, [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
            With multiplication nodes, [2,[5,3],[5,1],3] means besides the [2,5,5,3] KAN, there are 3 (1) mul nodes in layer 1 (2). 
        mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
        '''
        super(KANPDE, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.pde_order = pde_order
        self.pde_component = pde_component
        self.spatial_dim = spatial_dim
        self.dx = dx
        self.dy = dx if dy is None else dy
        self.is_periodic = is_periodic

        # layers_hidden = [self.pde_order + 1] + layers_hidden + [1]
        self.width = [(self.pde_order*self.spatial_dim+1) * self.pde_component] + layers_hidden + [self.pde_component]
        self.depth = len(self.width) - 1
        self.depth = len(self.width) - 1
        self.mult_arity = mult_arity

        for l in range(len(self.width)):
            if type(self.width[l]) == int:
                self.width[l] = [self.width[l], 0]

        width_in = self.width_in
        width_out = self.width_out

        self.layers = torch.nn.ModuleList()
        for l in range(self.depth):             
            self.layers.append(
                KANLinear(
                    width_in[l],
                    width_out[l+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, u: torch.Tensor, update_grid=False, nx=None, ny=None):
        # cache data
        self.cache_data = u
        self.acts = []

        if self.spatial_dim == 1:
            Du = [u]                                # u.shape = (ncomponent, nx)
            for _ in range(self.pde_order):
                out = compute_dudx(Du[-1], self.dx, is_periodic=self.is_periodic)
                Du.append(out)
            Du = torch.cat(Du, dim=0)               # Du.shape = (norder * ncomponent, nx)
            z = Du.permute([1,0])                   # z.shape = (nx, norder * ncomponent)
            self.acts.append(z.detach())

            for l in range(self.depth):  
                if update_grid:
                    self.layers[l].update_grid(z)
                z = self.layers[l](z)               # z.shape = (nx, ncomponent) 

                # multiplication
                dim_sum = self.width[l+1][0]
                dim_mult = self.width[l+1][1]
                z_mult = z[:,dim_sum::self.mult_arity] * z[:,dim_sum+1::self.mult_arity]
                for i in range(1, self.mult_arity-1):
                    z_mult = z_mult * z[:,dim_sum+i+1::self.mult_arity]

                if self.width[l+1][1] > 0:
                    z = torch.cat([z[:,:dim_sum], z_mult], dim=1)
                self.acts.append(z.detach())

            z = z.permute([1,0]) # z.shape = (ncomponent, nx) 

        if self.spatial_dim == 2:

            Du = [u]      
            Du.append(compute_dudx_2d(u, self.dx, is_periodic=self.is_periodic))
            Du.append(compute_dudy_2d(u, self.dy, is_periodic=self.is_periodic))
            # u.shape = (ncomponent, nx, ny)
            for _ in range(1, self.pde_order):
                out_x = compute_dudx_2d(Du[-2], self.dx, is_periodic=self.is_periodic)
                out_y = compute_dudy_2d(Du[-1], self.dy, is_periodic=self.is_periodic)
                Du.append(out_x)
                Du.append(out_y)

            Du = torch.cat(Du, dim=0)               # Du.shape = (norder * ncomponent * ndim, nx, ny)
            z = Du.permute([1,2,0])

            nx, ny, nd = z.shape
            z = z.view(-1, nd)

            self.acts.append(z.detach())

            for l in range(self.depth):  
                if update_grid:
                    self.layers[l].update_grid(z)
                z = self.layers[l](z)               # z.shape = (nx*ny, ncomponent) 

                # multiplication
                dim_sum = self.width[l+1][0]
                dim_mult = self.width[l+1][1]
                z_mult = z[...,dim_sum::self.mult_arity] * z[...,dim_sum+1::self.mult_arity]
                for i in range(1, self.mult_arity-1):
                    z_mult = z_mult * z[...,dim_sum+i+1::self.mult_arity]

                if self.width[l+1][1] > 0:
                    z = torch.cat([z[...,:dim_sum], z_mult], dim=-1)
                self.acts.append(z.detach())

            z = z.view(nx, ny, self.pde_component)

            z = z.permute([2,0,1])                    # z.shape = (ncomponent, nx*ny) 
        return z


    def regularization_loss(self, regularize_activation=1.0):
        weights = []
        for layer in self.layers:
            weights.append(layer.spline_weight.flatten())
        weights = torch.cat(weights, dim=-1)
        loss = torch.mean(torch.abs(weights)) * regularize_activation
        return loss

    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0]+width[l][1] for l in range(len(width))]
        return width_in
        
    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        width = self.width
        width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
        return width_out