<!-- preprocess -->
帮我写一段代码。

数据存储：data/raw路径下有所有种类的文件夹，例如annotation, depth, normal, mask, rgb, segm; 每个文件夹下面对应各个场景的数据，每一条数据的存储名称都是 {uid1}_{format}_{uid2}.ext的格式，例如mask文件夹下的其中一条数据是000588_mask_0030000.npy.gz；depth和mask和normal和segm的存储格式都是.npy.gz; annotation文件夹下是000588_annotation_0030000.json；我们定义新的每一个场景的uid为{uid1}-{uid2}。

代码要求：输出根目录为data/outputs/dataset。根目录下就是每一个场景的文件夹，每个文件夹以场景uid（即{uid1}-{uid2}）命名。遍历raw数据的depth和rgb和annotation，将它们复制到输出目录中名称对应的场景目录下，并将新文件分别命名为full_rgb.png 和 full_depth.npy 和 meta-data.json, 并再生成png格式的depth

python data_process.py --raw-root data/raw --output-root data/outputs/dataset --limit 20 


mask 是物体 full mask，segm 是 visible mask
原始data/raw目录下有一个segm文件夹，此文件为黑白图像，有36，357，0等数值
vis_mask_index = np.argwhere(vis_mask == obj_id)

向data_process中添加功能，对data/outputs/dataset下的每一个场景文件夹，把它分开成若干个物体文件夹。每个场景的annotation.json的"obj_dict"下有诸如“0” “1” “2”...这样的key，每个key对应的value也是一个字典，其中有物体的“obj_id”，它的value是一个列表，列表里的第0个值就是id的数值。每个物体文件夹以 场景uid-六位物体uid 命名，即 **{uid1}-{uid2}-{obj_id:6}**，仍然保存在data/outputs/dataset下。每个物体文件夹中，复制场景的full_depth.png, full_depth.npy, full_rgb.png, meta-data.json, 然后再从原目录data/raw下的segm目录中读取场景对应的segm文件（例如000588_segm_0030000.npy.gz），找出数值与obj_id相同的segm块，并把这块新的segm存储为黑白，命名为vis_mask.png


<!-- depth 对齐 -->

参考create_partial_pointcloud的从depth图片+物体的mask生成pointcloud的算法，参考data_process的输出数据格式，为每一个物体{uid1}-{uid2}-{obj_id:6}条目下，使用vis_mask和full_depth.npy和meta-data中的"camera_pose_tran" "camera_pose_rot" "camera_extrinsics" "camera_intrinsics"等数据，生成一个pointcloud 并以vis_pnts.ply格式保存。


python depth_to_pointcloud.py --dataset-root data/outputs/dataset --overwrite 


<!-- pc与scene对齐 -->
1. depth的计算：distance = depth * （far - near）+ near
这个数据集的depth的near和far是多少？可以从脚本中看出来吗？
不需要对齐，直接读取后就是正确的。

2. 如果ssr中的pose是blender格式，那么按照这种方式计算的点云结果是错误的。
必须变换成opencv格式的pose。
事实证明ssr dataset里的确是blender格式的pose。只需要翻转y和z即可。


<!-- 变换到3DFuture标准物体上 -->
<!-- 在metadata里，对于每一个物体， -->


在每个物体的metadata中，"obj_dict" 下的物体序号（如“1”号）下有"obj_tran"、"obj_rot"、"obj_scale"三个key，它们对应的value是物体转换到3DFUTURE原始资产的转换方式。转换之前的pointcloud保存为vis_pnts_scene.ply。帮我在depth_to_pointcloud中补充一个功能，按照这些obj2world的矩阵，把pointcloud再转换成原始资产形式，然后再保存为 vis_pnts.ply 和 vis_pnts.npy。


vis_pnts 与 raw model 对应
注意raw model 移动到blender里面会自动旋转90度 取消这个旋转就可以对上了


生成2D bbox
tar -czvf output_dataset.tar.gz data/outputs/dataset

参考 front3D_dataloader 和 data_process 代码, 帮我写一段生成2dbbox并可视化的代码. 给定文件夹根目录,下面每条物体数据文件夹{uid1}-{uid2}-{obj_id:6}内部都有一个metadata,metadata的 "obj_dict" 下面有 物体index (例如"0"或者"1"),物体index下面有"bbox_2d_from_3d";帮我可视化这个bbox2d, 读取full_rgb, 把整张图片变成白色,只有bbox内部是黑色的. 注意这个bbox 2d实际上是 8个3d bbox顶点到2d的投影,所以是8个点.需要根据这八个点的2d坐标找到正方体的投影.


python scripts/generate_amodal_mask.py --dataset-root data/outputs/test


# TODO
1. 有时间的话其实可以把 data_process 和 depth_to_pointcloud 封装成一个类 想要什么就挪什么










