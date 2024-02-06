import pulumi
import pulumi_kubernetes as k8s

# Assuming you've already configured the Pulumi Kubernetes provider
# to point to your existing or new Kubernetes cluster.

# Create a Kubernetes Namespace for organization and security (optional).
ai_model_ns = k8s.core.v1.Namespace("ai-model-ns")

# Define the AI model deployment.
ai_model_deployment = k8s.apps.v1.Deployment("ai-model-deployment",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        namespace=ai_model_ns.metadata["name"],
    ),
    spec=k8s.apps.v1.DeploymentSpecArgs(
        selector=k8s.meta.v1.LabelSelectorArgs(match_labels={"app": "ai-model"}),
        replicas=1,
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(labels={"app": "ai-model"}),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[k8s.core.v1.ContainerArgs(
                    name="model-container",
                    image="my-ai-model-image:latest",  # Replace this with your actual AI model container image.
                    ports=[k8s.core.v1.ContainerPortArgs(container_port=8080)],
                )],
            ),
        ),
    ),
)

# Expose the AI model as a service within the cluster.
ai_model_service = k8s.core.v1.Service("ai-model-service",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        namespace=ai_model_ns.metadata["name"],
    ),
    spec=k8s.core.v1.ServiceSpecArgs(
        selector={"app": "ai-model"},
        ports=[k8s.core.v1.ServicePortArgs(port=8080)],
    ),
)

# Define the Envoy deployment to handle TLS termination.
envoy_deployment = k8s.apps.v1.Deployment("envoy-deployment",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        namespace=ai_model_ns.metadata["name"],
    ),
    spec=k8s.apps.v1.DeploymentSpecArgs(
        selector=k8s.meta.v1.LabelSelectorArgs(match_labels={"app": "envoy"}),
        replicas=1,
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(labels={"app": "envoy"}),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[k8s.core.v1.ContainerArgs(
                    name="envoy-container",
                    image="envoyproxy/envoy:v1.18.3",  # Replace this with the desired Envoy image.
                    ports=[k8s.core.v1.ContainerPortArgs(container_port=443)],
                    volumeMounts=[k8s.core.v1.VolumeMountArgs(
                        name="tls-certs",
                        mount_path="/etc/tls",
                        read_only=True,
                    )],
                    args=[
                      "-c", "/etc/envoy/envoy-config.yaml",  # Point this to your actual Envoy config file.
                      "--log-level", "info",
                    ],
                )],
                volumes=[k8s.core.v1.VolumeArgs(
                    name="tls-certs",
                    secret=k8s.core.v1.SecretVolumeSourceArgs(
                        secret_name="tls-certs-secret",  # This secret should contain your TLS certificate and key.
                    ),
                )],
            ),
        ),
    ),
)

# Expose Envoy as a service to the outside world.
envoy_service = k8s.core.v1.Service("envoy-service",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        namespace=ai_model_ns.metadata["name"],
    ),
    spec=k8s.core.v1.ServiceSpecArgs(
        type="LoadBalancer",  # This will create a cloud load balancer to expose Envoy externally.
        selector={"app": "envoy"},
        ports=[k8s.core.v1.ServicePortArgs(port=443, target_port=443)],
    ),
)

# Export the Envoy service endpoint.
pulumi.export("envoy_service_endpoint", envoy_service.status["load_balancer"]["ingress"][0]["ip"])