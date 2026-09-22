# Kubernetes deployment

These manifests provide a cloud-neutral production baseline. They expect a published `support-search` image,
a PostgreSQL database reachable from the cluster, a metrics server for the HPA, and a Secret named
`support-search-secrets`.

Before deployment:

1. Replace `support-search:latest` with an immutable registry image and digest.
2. Create `support-search-secrets` with `SEARCH_DATABASE_URL` and `SEARCH_API_KEY`.
3. Apply database migrations using the release pipeline.
4. Add a platform-specific Ingress/Gateway with TLS and rate limiting.
5. Adjust CPU, memory, replica, and autoscaling values using measured load-test results.

```bash
kubectl apply -k ops/kubernetes
kubectl rollout status deployment/support-search
kubectl get deploy,pods,svc,hpa,pdb
```

Use an environment-specific Kustomize overlay in a real deployment rather than editing these baseline files.
