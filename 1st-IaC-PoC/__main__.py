import pulumi
import pulumi_azure_native as azure_native
from pulumi_azure_native import resources
from pulumi_azure_native import operationalinsights
from pulumi_azure_native import securityinsights
from pulumi_azure_native import containerservice

# Creating a resource group for all resources
resource_group = resources.ResourceGroup('rg')

# Deploying ARM template for Enterprise-Scale Landing Zone
arm_template = azure_native.resources.Deployment(
    'eslzDeployment',
    properties=azure_native.resources.DeploymentPropertiesArgs(
        mode='Incremental',
        template_link=azure_native.resources.TemplateLinkArgs(
            uri='https://github.com/Azure/Enterprise-Scale/blob/main/docs/reference/treyresearch/armTemplates/mainTemplate.json',
            # Assuming this URI is publicly accessible and does not require authentication; otherwise, use SAS token
        )
    ),
    resource_group_name=resource_group.name
)

# Deploying Azure Sentinel
sentinel_workspace = operationalinsights.Workspace(
    'sentinelWorkspace',
    resource_group_name=resource_group.name,
    sku=operationalinsights.WorkspaceSkuArgs(name='PerGB2018')
)

sentinel = securityinsights.Solution(
    'azureSentinel',
    resource_group_name=resource_group.name,
    workspace_name=sentinel_workspace.name,
    product='Azure Sentinel'
)

# Creating Kubernetes confidential clusters for scaling
k8s_cluster = containerservice.ManagedCluster(
    'confidentialCluster',
    resource_group_name=resource_group.name,
    agent_pool_profiles=[containerservice.ManagedClusterAgentPoolProfileArgs(
        count=3,
        max_pods=110,
        mode='System',
        name='agentpool',
        node_labels={'type': 'confidential'},
        os_disk_size_gb=128,
        os_type='Linux',
        vm_size='Standard_DC8_v2',  # Size of VMs in the node pool supporting confidential computing
    )],
    enable_rbac=True,
    identity=containerservice.ManagedClusterIdentityArgs(
        type='SystemAssigned'
    ),
    kubernetes_version='1.18.14',
    sku=containerservice.ManagedClusterSKUArgs(
        name='Basic',
        tier='Free'
    )
)

# Export the outputs
pulumi.export('resource_group_name', resource_group.name)
pulumi.export('sentinel_workspace_id', sentinel_workspace.id)
pulumi.export('sentinel_solution_id', sentinel.id)
pulumi.export('k8s_cluster_id', k8s_cluster.id)