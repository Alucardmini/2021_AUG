??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
#
	LogicalOr
x

y

z
?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58??

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
dtype0	*
_output_shapes
: *
shape: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
^
SexPlaceholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
c
EmbarkedPlaceholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
a
PclassPlaceholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
^
AgePlaceholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
i
input_layer/Age/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 

input_layer/Age/ExpandDims
ExpandDimsAgeinput_layer/Age/ExpandDims/dim*
T0*'
_output_shapes
:?????????
_
input_layer/Age/ShapeShapeinput_layer/Age/ExpandDims*
T0*
_output_shapes
:
m
#input_layer/Age/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%input_layer/Age/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
o
%input_layer/Age/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
input_layer/Age/strided_sliceStridedSliceinput_layer/Age/Shape#input_layer/Age/strided_slice/stack%input_layer/Age/strided_slice/stack_1%input_layer/Age/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
a
input_layer/Age/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
input_layer/Age/Reshape/shapePackinput_layer/Age/strided_sliceinput_layer/Age/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
input_layer/Age/ReshapeReshapeinput_layer/Age/ExpandDimsinput_layer/Age/Reshape/shape*'
_output_shapes
:?????????*
T0
x
-input_layer/Embarked_indicator/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
)input_layer/Embarked_indicator/ExpandDims
ExpandDimsEmbarked-input_layer/Embarked_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
~
=input_layer/Embarked_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
7input_layer/Embarked_indicator/to_sparse_input/NotEqualNotEqual)input_layer/Embarked_indicator/ExpandDims=input_layer/Embarked_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
6input_layer/Embarked_indicator/to_sparse_input/indicesWhere7input_layer/Embarked_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
5input_layer/Embarked_indicator/to_sparse_input/valuesGatherNd)input_layer/Embarked_indicator/ExpandDims6input_layer/Embarked_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
:input_layer/Embarked_indicator/to_sparse_input/dense_shapeShape)input_layer/Embarked_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
%input_layer/Embarked_indicator/lookupStringToHashBucketFast5input_layer/Embarked_indicator/to_sparse_input/values*
num_buckets*#
_output_shapes
:?????????
?
:input_layer/Embarked_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
,input_layer/Embarked_indicator/SparseToDenseSparseToDense6input_layer/Embarked_indicator/to_sparse_input/indices:input_layer/Embarked_indicator/to_sparse_input/dense_shape%input_layer/Embarked_indicator/lookup:input_layer/Embarked_indicator/SparseToDense/default_value*
Tindices0	*
T0	*'
_output_shapes
:?????????
q
,input_layer/Embarked_indicator/one_hot/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
s
.input_layer/Embarked_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
n
,input_layer/Embarked_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
t
/input_layer/Embarked_indicator/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
0input_layer/Embarked_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
&input_layer/Embarked_indicator/one_hotOneHot,input_layer/Embarked_indicator/SparseToDense,input_layer/Embarked_indicator/one_hot/depth/input_layer/Embarked_indicator/one_hot/on_value0input_layer/Embarked_indicator/one_hot/off_value*+
_output_shapes
:?????????*
T0
?
4input_layer/Embarked_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
"input_layer/Embarked_indicator/SumSum&input_layer/Embarked_indicator/one_hot4input_layer/Embarked_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
v
$input_layer/Embarked_indicator/ShapeShape"input_layer/Embarked_indicator/Sum*
T0*
_output_shapes
:
|
2input_layer/Embarked_indicator/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
~
4input_layer/Embarked_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4input_layer/Embarked_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
,input_layer/Embarked_indicator/strided_sliceStridedSlice$input_layer/Embarked_indicator/Shape2input_layer/Embarked_indicator/strided_slice/stack4input_layer/Embarked_indicator/strided_slice/stack_14input_layer/Embarked_indicator/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
p
.input_layer/Embarked_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
,input_layer/Embarked_indicator/Reshape/shapePack,input_layer/Embarked_indicator/strided_slice.input_layer/Embarked_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
&input_layer/Embarked_indicator/ReshapeReshape"input_layer/Embarked_indicator/Sum,input_layer/Embarked_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
v
+input_layer/Pclass_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
'input_layer/Pclass_indicator/ExpandDims
ExpandDimsPclass+input_layer/Pclass_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
;input_layer/Pclass_indicator/to_sparse_input/ignore_value/xConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
5input_layer/Pclass_indicator/to_sparse_input/NotEqualNotEqual'input_layer/Pclass_indicator/ExpandDims;input_layer/Pclass_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
4input_layer/Pclass_indicator/to_sparse_input/indicesWhere5input_layer/Pclass_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
3input_layer/Pclass_indicator/to_sparse_input/valuesGatherNd'input_layer/Pclass_indicator/ExpandDims4input_layer/Pclass_indicator/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
8input_layer/Pclass_indicator/to_sparse_input/dense_shapeShape'input_layer/Pclass_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
#input_layer/Pclass_indicator/valuesCast3input_layer/Pclass_indicator/to_sparse_input/values*

SrcT0*#
_output_shapes
:?????????*

DstT0	
l
*input_layer/Pclass_indicator/num_buckets/xConst*
value	B :
*
dtype0*
_output_shapes
: 
?
(input_layer/Pclass_indicator/num_bucketsCast*input_layer/Pclass_indicator/num_buckets/x*
_output_shapes
: *

DstT0	*

SrcT0
e
#input_layer/Pclass_indicator/zero/xConst*
value	B : *
dtype0*
_output_shapes
: 
~
!input_layer/Pclass_indicator/zeroCast#input_layer/Pclass_indicator/zero/x*
_output_shapes
: *

DstT0	*

SrcT0
?
!input_layer/Pclass_indicator/LessLess#input_layer/Pclass_indicator/values!input_layer/Pclass_indicator/zero*#
_output_shapes
:?????????*
T0	
?
)input_layer/Pclass_indicator/GreaterEqualGreaterEqual#input_layer/Pclass_indicator/values(input_layer/Pclass_indicator/num_buckets*
T0	*#
_output_shapes
:?????????
?
)input_layer/Pclass_indicator/out_of_range	LogicalOr!input_layer/Pclass_indicator/Less)input_layer/Pclass_indicator/GreaterEqual*#
_output_shapes
:?????????
u
"input_layer/Pclass_indicator/ShapeShape#input_layer/Pclass_indicator/values*
T0	*
_output_shapes
:
e
#input_layer/Pclass_indicator/Cast/xConst*
value	B : *
dtype0*
_output_shapes
: 
~
!input_layer/Pclass_indicator/CastCast#input_layer/Pclass_indicator/Cast/x*

SrcT0*
_output_shapes
: *

DstT0	
?
+input_layer/Pclass_indicator/default_valuesFill"input_layer/Pclass_indicator/Shape!input_layer/Pclass_indicator/Cast*#
_output_shapes
:?????????*
T0	
?
#input_layer/Pclass_indicator/SelectSelect)input_layer/Pclass_indicator/out_of_range+input_layer/Pclass_indicator/default_values#input_layer/Pclass_indicator/values*#
_output_shapes
:?????????*
T0	
?
8input_layer/Pclass_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
*input_layer/Pclass_indicator/SparseToDenseSparseToDense4input_layer/Pclass_indicator/to_sparse_input/indices8input_layer/Pclass_indicator/to_sparse_input/dense_shape#input_layer/Pclass_indicator/Select8input_layer/Pclass_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
Tindices0	*
T0	
o
*input_layer/Pclass_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
q
,input_layer/Pclass_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
l
*input_layer/Pclass_indicator/one_hot/depthConst*
value	B :
*
dtype0*
_output_shapes
: 
r
-input_layer/Pclass_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
.input_layer/Pclass_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
$input_layer/Pclass_indicator/one_hotOneHot*input_layer/Pclass_indicator/SparseToDense*input_layer/Pclass_indicator/one_hot/depth-input_layer/Pclass_indicator/one_hot/on_value.input_layer/Pclass_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????

?
2input_layer/Pclass_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
 input_layer/Pclass_indicator/SumSum$input_layer/Pclass_indicator/one_hot2input_layer/Pclass_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????

t
$input_layer/Pclass_indicator/Shape_1Shape input_layer/Pclass_indicator/Sum*
T0*
_output_shapes
:
z
0input_layer/Pclass_indicator/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
|
2input_layer/Pclass_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2input_layer/Pclass_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
*input_layer/Pclass_indicator/strided_sliceStridedSlice$input_layer/Pclass_indicator/Shape_10input_layer/Pclass_indicator/strided_slice/stack2input_layer/Pclass_indicator/strided_slice/stack_12input_layer/Pclass_indicator/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
n
,input_layer/Pclass_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

?
*input_layer/Pclass_indicator/Reshape/shapePack*input_layer/Pclass_indicator/strided_slice,input_layer/Pclass_indicator/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
$input_layer/Pclass_indicator/ReshapeReshape input_layer/Pclass_indicator/Sum*input_layer/Pclass_indicator/Reshape/shape*'
_output_shapes
:?????????
*
T0
s
(input_layer/Sex_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
$input_layer/Sex_indicator/ExpandDims
ExpandDimsSex(input_layer/Sex_indicator/ExpandDims/dim*'
_output_shapes
:?????????*
T0
y
8input_layer/Sex_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
2input_layer/Sex_indicator/to_sparse_input/NotEqualNotEqual$input_layer/Sex_indicator/ExpandDims8input_layer/Sex_indicator/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
1input_layer/Sex_indicator/to_sparse_input/indicesWhere2input_layer/Sex_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
0input_layer/Sex_indicator/to_sparse_input/valuesGatherNd$input_layer/Sex_indicator/ExpandDims1input_layer/Sex_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
5input_layer/Sex_indicator/to_sparse_input/dense_shapeShape$input_layer/Sex_indicator/ExpandDims*
_output_shapes
:*
T0*
out_type0	
?
 input_layer/Sex_indicator/lookupStringToHashBucketFast0input_layer/Sex_indicator/to_sparse_input/values*
num_buckets*#
_output_shapes
:?????????
?
5input_layer/Sex_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
'input_layer/Sex_indicator/SparseToDenseSparseToDense1input_layer/Sex_indicator/to_sparse_input/indices5input_layer/Sex_indicator/to_sparse_input/dense_shape input_layer/Sex_indicator/lookup5input_layer/Sex_indicator/SparseToDense/default_value*
Tindices0	*
T0	*'
_output_shapes
:?????????
l
'input_layer/Sex_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
n
)input_layer/Sex_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
i
'input_layer/Sex_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
o
*input_layer/Sex_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
p
+input_layer/Sex_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
!input_layer/Sex_indicator/one_hotOneHot'input_layer/Sex_indicator/SparseToDense'input_layer/Sex_indicator/one_hot/depth*input_layer/Sex_indicator/one_hot/on_value+input_layer/Sex_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????
?
/input_layer/Sex_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
input_layer/Sex_indicator/SumSum!input_layer/Sex_indicator/one_hot/input_layer/Sex_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
l
input_layer/Sex_indicator/ShapeShapeinput_layer/Sex_indicator/Sum*
T0*
_output_shapes
:
w
-input_layer/Sex_indicator/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/input_layer/Sex_indicator/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/input_layer/Sex_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
'input_layer/Sex_indicator/strided_sliceStridedSliceinput_layer/Sex_indicator/Shape-input_layer/Sex_indicator/strided_slice/stack/input_layer/Sex_indicator/strided_slice/stack_1/input_layer/Sex_indicator/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
k
)input_layer/Sex_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
'input_layer/Sex_indicator/Reshape/shapePack'input_layer/Sex_indicator/strided_slice)input_layer/Sex_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
!input_layer/Sex_indicator/ReshapeReshapeinput_layer/Sex_indicator/Sum'input_layer/Sex_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
Y
input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
input_layer/concatConcatV2input_layer/Age/Reshape&input_layer/Embarked_indicator/Reshape$input_layer/Pclass_indicator/Reshape!input_layer/Sex_indicator/Reshapeinput_layer/concat/axis*
N*'
_output_shapes
:?????????*
T0
?
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *g{??*
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *g{?=*
_class
loc:@dense/kernel
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	?
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?*
T0*
_class
loc:@dense/kernel
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	?

dense/kernel
VariableV2*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	?*
shape:	?
?
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	?
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	?*
T0*
_class
loc:@dense/kernel
?
,dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:?*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
?
"dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
?
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*
T0*
_class
loc:@dense/bias*
_output_shapes	
:?
s

dense/bias
VariableV2*
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:?*
shape:?
?
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_output_shapes	
:?*
T0*
_class
loc:@dense/bias
l
dense/bias/readIdentity
dense/bias*
_output_shapes	
:?*
T0*
_class
loc:@dense/bias
p
dense/MatMulMatMulinput_layer/concatdense/kernel/read*(
_output_shapes
:??????????*
T0
j
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*(
_output_shapes
:??????????
^
dropout/IdentityIdentitydense/BiasAdd*
T0*(
_output_shapes
:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
?
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ܰ??*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ܰ?=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	?
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
:	?*
T0*!
_class
loc:@dense_1/kernel
?
dense_1/kernel
VariableV2*
shape:	?*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	?
?
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
?
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias
u
dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:*
shape:
?
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
q
dense_1/MatMulMatMuldropout/Identitydense_1/kernel/read*
T0*'
_output_shapes
:?????????
o
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:?????????*
T0
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
a
ArgMaxArgMaxdense_1/BiasAddArgMax/dimension*
T0*#
_output_shapes
:?????????

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_2373ad6424aa4704a3cbf34fe9c10d2d/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*X
valueOBMB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:
|
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step"/device:CPU:0*
dtypes	
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*X
valueOBMB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2	
v
save/AssignAssign
dense/biassave/RestoreV2*
T0*
_class
loc:@dense/bias*
_output_shapes	
:?
?
save/Assign_1Assigndense/kernelsave/RestoreV2:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	?
}
save/Assign_2Assigndense_1/biassave/RestoreV2:2*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
?
save/Assign_3Assigndense_1/kernelsave/RestoreV2:3*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	?
w
save/Assign_4Assignglobal_stepsave/RestoreV2:4*
_output_shapes
: *
T0	*
_class
loc:@global_step
h
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
trainable_variables??
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"?
	variables??
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"%
saved_model_main_op


group_deps*?
serving_default?

Age
Age:0?????????
%
Pclass
Pclass:0?????????
)
Embarked

Embarked:0?????????

Sex
Sex:0?????????%
result
ArgMax:0	?????????tensorflow/serving/predict