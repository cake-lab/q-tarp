

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class StandardTransformerRuntimeModel(torch.nn.Module):
    """Graph + global-feature model for predicting circuit execution time."""

    def __init__(self, args, length_of_x: int = 178, length_of_gf: int = 41):
        # Initialize the parent torch.nn.Module.
        super().__init__()
        self.args = args
        self.length_of_x = int(length_of_x)
        self.length_of_gf = int(length_of_gf)
        
        self.use_graph_features = bool(getattr(args, "use_graph_features", True))
        self.use_global_features = bool(getattr(args, "use_global_features", True))
        
        self.graph_hidden_dim = int(getattr(args, "graph_hidden_dim", self.length_of_x))
        self.global_hidden_dim = int(getattr(args, "global_hidden_dim", 64))
        
        self.fusion_hidden_dim1 = int(getattr(args, "fusion_hidden_dim1", 512))
        self.fusion_hidden_dim2 = int(getattr(args, "fusion_hidden_dim2", 512))
        self.fusion_hidden_dim3 = int(getattr(args, "fusion_hidden_dim3", 128))
        
        self.num_layers = int(getattr(args, "num_layers", 3))
        # Read dropout probability, defaulting to 0.0 so old behavior is preserved.
        self.dropout = float(getattr(args, "dropout", 0.0))
        self.debug_trace = bool(getattr(args, "debug_trace", False))
        self.debug_max_forwards = int(getattr(args, "debug_max_forwards", 2))
        self._debug_forward_calls = 0
        self._debug_active = False

        feature_mask = self._build_feature_mask(args)
        # Register the mask as a buffer so it moves with model.to(device).
        self.register_buffer("feature_mask", feature_mask)
        # Count how many node features remain after masking.
        masked_x_dim = int(feature_mask.sum().item())

        if self.use_graph_features:
            # ModuleList is the standard PyTorch container for repeated layers.
            self.convs = torch.nn.ModuleList()
            self.convs.append(TransformerConv(masked_x_dim, self.graph_hidden_dim))
            # Additional TransformerConv layers keep the same hidden dimension.
            for _ in range(1, self.num_layers):
                self.convs.append(TransformerConv(self.graph_hidden_dim, self.graph_hidden_dim))
        else:
            self.convs = torch.nn.ModuleList()

        # Create global feature MLP only if global features are enabled.
        if self.use_global_features:
            self.global_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.length_of_gf, self.global_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                
                torch.nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
                torch.nn.ReLU(),
            )

        # Compute the feature dimension entering the fusion MLP.
        fusion_input_dim = self._fusion_input_dim()
        
        
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_dim, self.fusion_hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.fusion_hidden_dim1, self.fusion_hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.fusion_hidden_dim2, self.fusion_hidden_dim3),
            torch.nn.ReLU(),
            
            torch.nn.Linear(self.fusion_hidden_dim3, 1),
        )

    def _debug_print(self, label, tensor, max_rows: int = 2, max_cols: int = 8):
        """Print a concise tensor preview when debug tracing is enabled."""
        if not self._debug_active:
            return
        detached = tensor.detach().cpu()
        if detached.dim() == 0:
            preview = float(detached.item())
        elif detached.dim() == 1:
            preview = detached[:max_cols].tolist()
        else:
            preview = detached[:max_rows, :max_cols].tolist()
        print(f"[model-debug] {label} shape={tuple(detached.shape)} preview={preview}", flush=True)

    def _build_feature_mask(self, args):
        """Build a boolean mask for selected node-feature groups."""
        # Start with all 178 original node features enabled.
        mask = torch.ones(self.length_of_x, dtype=torch.bool)
        # Disable gate-type one-hot features if requested.
        if not bool(getattr(args, "use_gate_type", True)):
            mask[0:46] = False
        # Disable qubit-index indicator features if requested.
        if not bool(getattr(args, "use_qubit_index", True)):
            mask[46:173] = False
        # Disable T1/T2 noise features if requested.
        if not bool(getattr(args, "use_T1T2", True)):
            mask[173:177] = False
        # Disable gate-index feature if requested.
        if not bool(getattr(args, "use_gate_index", True)):
            mask[177] = False
        return mask

    def _fusion_input_dim(self):
        """Calculate the input dimension of the fusion MLP."""
        dim = 0
        # Add graph embedding size when the graph branch is active.
        if self.use_graph_features:
            dim += self.graph_hidden_dim
            
        # Add global embedding size when the global branch is active.
        if self.use_global_features:
            dim += self.global_hidden_dim
        return dim

    def _encode_graph(self, data):
        """Encode graph node features into one graph-level embedding."""
        x = data.x.float()
        x = x[:, self.feature_mask]
        self._debug_print("graph/x_masked", x)
        edge_index = data.edge_index
        # Pass node features through each TransformerConv layer.
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            # Apply dropout during training only if dropout > 0.
            x = F.dropout(x, p=self.dropout, training=self.training)
            self._debug_print(f"graph/conv_{layer_idx}_out", x)
        pooled = global_mean_pool(x, data.batch)
        self._debug_print("graph/pooled", pooled)
        return pooled

    def _encode_global(self, data):
        """Encode global circuit features into one hidden vector per graph."""
        gf = data.global_features.float()
        # Make sure global features are shaped [batch_size, num_features].
        gf = gf.view(gf.size(0), -1)
        self._debug_print("global/input", gf)
        encoded = self.global_mlp(gf)
        self._debug_print("global/encoded", encoded)
        return encoded

    def forward(self, data):
        """Run a forward pass and return one prediction per graph."""
        self._debug_active = self.debug_trace and self._debug_forward_calls < self.debug_max_forwards
        if self._debug_active:
            print(
                f"[model-debug] forward_call={self._debug_forward_calls + 1} "
                f"num_graphs={int(data.num_graphs)} x_shape={tuple(data.x.shape)} "
                f"edge_index_shape={tuple(data.edge_index.shape)} "
                f"global_features_shape={tuple(data.global_features.shape)}",
                flush=True,
            )
        embeddings = []
        if self.use_graph_features:
            # Encode the graph branch and append its graph-level embedding.
            embeddings.append(self._encode_graph(data))
        if self.use_global_features:
            # Encode the global feature branch and append its embedding.
            embeddings.append(self._encode_global(data))
        # Concatenate active branch embeddings along feature dimension.
        x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        self._debug_print("fusion/input", x)
        x = self.fusion_mlp(x)
        self._debug_print("fusion/output_raw", x)
        # Remove the final singleton dimension so output shape is [batch_size].
        output = x.squeeze(-1)
        self._debug_print("output", output)
        if self._debug_active:
            print("[model-debug] forward end", flush=True)
        self._debug_forward_calls += 1
        self._debug_active = False
        return output
