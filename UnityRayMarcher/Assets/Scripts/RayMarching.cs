using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;


public class RayMarching : MonoBehaviour
{
    public enum LayerType
    {
        FourierFeatures = 0,
        Dense = 1,
        Vector = 2,
    }

    struct Layer
    {
        public uint startIndex;
        public uint Rows;
        public uint Columns;
        public LayerType Type;
    }

    struct Matrix3x4
    {
        public Vector4 m0, m1, m2;
        public Vector4 this[int i]
        {
            get
            {
                switch (i)
                {
                    case 0:
                        return m0;
                    case 1:
                        return m1;
                    case 2:
                        return m2;
                    default:
                        return Vector4.zero;
                }
            }
            set
            {
                switch (i)
                {
                    case 0:
                        m0 = value;
                        break;
                    case 1:
                        m1 = value;
                        break;
                    case 2:
                        m2 = value;
                        break;
                    default:
                        Debug.LogError("Index out of range for Matrix3x4");
                        break;
                }
            }
        }
    }

    struct Matrix3x4Shader
    {
        public Vector3 m0, m1, m2, m3;
        public Vector3 this[int i]
        {
            get
            {
                switch (i)
                {
                    case 0:
                        return m0;
                    case 1:
                        return m1;
                    case 2:
                        return m2;
                    case 3:
                        return m3;
                    default:
                        return Vector4.zero;
                }
            }
            set
            {
                switch (i)
                {
                    case 0:
                        m0 = value;
                        break;
                    case 1:
                        m1 = value;
                        break;
                    case 2:
                        m2 = value;
                        break;
                    case 3:
                        m3 = value;
                        break;
                    default:
                        Debug.LogError("Index out of range for Matrix3x4");
                        break;
                }
            }
        }
    }


    public ComputeShader computeShader;

    public Vector4 Sphere = new Vector4(0, 2, 6, 1);
    public Vector4 Light = new Vector4(1, 5, 1, 1000);
    public float PlaneHeight = 0;
    public float ShadowIntensity = 2.0f;
    public Texture Skybox;

    private Camera _camera;
    private RenderTexture _texture;
    private Vector2 _iResolution;
    private int _mainKernelIndex;

    private ComputeBuffer _mat3x4Buffer;
    private ComputeBuffer _mat4x4Buffer;
    private ComputeBuffer _float4Buffer;

    private List<Matrix3x4Shader> _matrix3X4s;
    private List<Matrix4x4> _matrix4X4s;
    private List<Vector4> _float4s;

    private List<Layer> _layers;

    private ComputeBuffer _layersBuffer;

    private uint _mat3x4_next = 0;
    private uint _mat4x4_next = 0;
    private uint _float4_next = 0;

    private bool _layersUpdated = false;

    public void Start()
    {
        _camera = GetComponent<Camera>();
        _texture = new RenderTexture(_camera.pixelWidth, _camera.pixelHeight, 24);
        //_texture = new RenderTexture(512, 256, 24);

        _texture.enableRandomWrite = true;
        _texture.Create();
        _iResolution = new Vector2(_texture.width, _texture.height);
        _mainKernelIndex = computeShader.FindKernel("CSMain");

        _layers = new List<Layer>();
        _matrix3X4s = new List<Matrix3x4Shader>();
        _matrix4X4s = new List<Matrix4x4>();
        _float4s = new List<Vector4>();

        _mat3x4Buffer = new ComputeBuffer(64, sizeof(float) * 12);
        _mat4x4Buffer = new ComputeBuffer(64, sizeof(float) * 16);
        _float4Buffer = new ComputeBuffer(16, sizeof(float) * 4);
        _layersBuffer = new ComputeBuffer(64, sizeof(int) * 4);
        ReadWeights("C:\\Users\\LAPTOP\\Documents\\DNN-SDF-Approximation\\NN Models\\Alexander_FF32_D16_D16_D1_weights.unity.txt", 16, 2, 16);
        Debug.Log(_layers.Count);
        Debug.Log(_matrix3X4s.Count);
        Debug.Log(_matrix4X4s.Count);
        Debug.Log(_float4s.Count);

    }

    public void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (_texture == null ||
            _matrix3X4s == null ||
            _matrix4X4s == null ||
            _float4s == null)
        {
            return;
        }

        if (_layersUpdated)
        {
            _mat3x4Buffer.SetData(_matrix3X4s);
            _mat4x4Buffer.SetData(_matrix4X4s);
            _float4Buffer.SetData(_float4s);
            _layersBuffer.SetData(_layers);

            computeShader.SetBuffer(_mainKernelIndex, "Layers", _layersBuffer);
            computeShader.SetBuffer(_mainKernelIndex, "Mat3x4s", _mat3x4Buffer);
            computeShader.SetBuffer(_mainKernelIndex, "Mat4x4s", _mat4x4Buffer);
            computeShader.SetBuffer(_mainKernelIndex, "Float4s", _float4Buffer);


            _layersUpdated = false;
        }

        computeShader.SetTexture(0, "Result", _texture);
        computeShader.SetTexture(0, "Skybox", Skybox);
        computeShader.SetVector("iResolution", _iResolution);
        computeShader.SetVector("SPHERE", Sphere);
        computeShader.SetVector("LIGHT", Light);
        computeShader.SetFloat("PLANE_HEIGHT", PlaneHeight);
        computeShader.SetFloat("ShadowIntensity", ShadowIntensity);
        computeShader.SetInt("LayersCount", _layers.Count);

        computeShader.SetMatrix("CameraTransform", _camera.transform.localToWorldMatrix);
        computeShader.Dispatch(_mainKernelIndex, (int)(_iResolution.x / 32), (int)(_iResolution.y / 32), 1);
        Graphics.Blit(_texture, destination);
    }

    public void AddLayer(List<float> data, int rows, int columns, LayerType type)
    {
        switch (type)
        {
            case LayerType.FourierFeatures:
                AddFourierFeaturesLayer(data.Take(rows * columns).ToList(), rows, columns);
                break;
            case LayerType.Dense:
                AddDenseLayer(data.Take(rows * columns).ToList(), rows, columns);
                break;
            case LayerType.Vector:
                AddVectorLayer(data.Take(rows * columns).ToList(), rows);
                break;
            default:
                Debug.LogError("Unindentified Layer Type");
                break;
        }
    }

    public void AddFourierFeaturesLayer(List<float> data, int rows, int columns)
    {
        if (rows != 3)
        {
            Debug.LogError("Fourier Features layer should be 3xN in shape.");
            return;
        }

        if (data.Count != rows * columns)
        {
            Debug.LogError("Data count mismatch! the number of values given doesn't match the given dimensions.");
            return;
        }


        for (int i = 0; i < (4 - (columns % 4)) % 4; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                data.Add(0);
            }
        }

        int matrixCount = columns / 4;
        Matrix3x4[] matrices = new Matrix3x4[matrixCount];


        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixCount; j++)
            {
                var x = data.Skip(i * columns + j * 4).Take(4).ToArray();
                matrices[j][i] = new Vector4(x[0], x[1], x[2], x[3]);
            }
        }

        Layer layer = new Layer
        {
            startIndex = _mat3x4_next,
            Rows = (uint)rows,
            Columns = (uint)columns,
            Type = LayerType.FourierFeatures,
        };

        foreach (var mat in matrices)
        {
            Matrix3x4Shader tmp = default(Matrix3x4Shader);
            tmp.m0 = new Vector3(mat[0].x, mat[1].x, mat[2].x);
            tmp.m1 = new Vector3(mat[0].y, mat[1].y, mat[2].y);
            tmp.m2 = new Vector3(mat[0].z, mat[1].z, mat[2].z);
            tmp.m3 = new Vector3(mat[0].w, mat[1].w, mat[2].w);

            _matrix3X4s.Add(tmp);
            _mat3x4_next++;
        }

        _layers.Add(layer);
    }

    public void AddDenseLayer(List<float> data, int rows, int columns)
    {
        if (data.Count != rows * columns)
        {
            Debug.LogError("Data count mismatch! the number of values given doesn't match the given dimensions.");
            return;
        }

        float GetElement(int i, int j)
        {
            int index = i * columns + j;
            if (index >= data.Count)
            {
                return 0;
            }
            return data[index];
        }

        List<Matrix4x4> matrices = new List<Matrix4x4>();

        for (int i = 0; i < rows; i += 4)
        {
            for (int j = 0; j < columns; j += 4)
            {
                Matrix4x4 m = default(Matrix4x4);
                for (int r = i; r < i + 4; r++)
                {
                    Vector4 v = new Vector4(GetElement(r, j),
                                            GetElement(r, j + 1),
                                            GetElement(r, j + 2),
                                            GetElement(r, j + 3));
                    m.SetRow(r - i, v);
                }
                matrices.Add(m);
            }
        }

        Layer layer = new Layer
        {
            startIndex = _mat4x4_next,
            Rows = (uint)rows,
            Columns = (uint)columns,
            Type = LayerType.Dense,
        };

        foreach (var mat in matrices)
        {
            _matrix4X4s.Add(mat);
            _mat4x4_next++;
        }

        _layers.Add(layer);
    }

    public void AddVectorLayer(List<float> data, int n)
    {
        if (data.Count != n)
        {
            Debug.LogError("The number of elements doesn't match the size given.");
            return;
        }

        float GetElement(int i)
        {
            return i >= data.Count ? 0 : data[i];
        }

        List<Vector4> vectors = new List<Vector4>();
        for (int i = 0; i < n; i += 4)
        {
            vectors.Add(new Vector4(GetElement(i),
                                    GetElement(i + 1),
                                    GetElement(i + 2),
                                    GetElement(i + 3)));
        }

        Layer layer = new Layer
        {
            startIndex = _float4_next,
            Rows = (uint)n,
            Columns = 1,
            Type = LayerType.Vector,
        };

        foreach (var vec in vectors)
        {
            _float4s.Add(vec);
            _float4_next++;
        }

        _layers.Add(layer);
    }

    public void ReadWeights(string path, int fourierMappingsCount, int hiddenLayerCount, int hiddenLayerSize)
    {
        string[] lines = File.ReadAllLines(path);
        if (lines.Length != hiddenLayerCount + 2)
        {
            Debug.LogError("Invalid weights data given.");
            return;
        }

        //Fourier Mappings
        List<float> ff = lines[0]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .ToList();
        AddLayer(ff, 3, fourierMappingsCount, LayerType.FourierFeatures);

        // First Dense Layer
        List<float> d1 = lines[1]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Take(fourierMappingsCount * 2 * hiddenLayerSize)
                            .ToList();
        AddLayer(d1, hiddenLayerSize, fourierMappingsCount * 2, LayerType.Dense);

        // First Layer Bias
        List<float> b1 = lines[1]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Skip(fourierMappingsCount * 2 * hiddenLayerSize)
                            .ToList();

        AddLayer(b1, b1.Count, 1, LayerType.Vector);

        // Second Dense Layer
        List<float> d2 = lines[2]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Take(hiddenLayerSize * hiddenLayerSize)
                            .ToList();
        AddLayer(d2, hiddenLayerSize, hiddenLayerSize, LayerType.Dense);

        // Second Layer Bias
        List<float> b2 = lines[2]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Skip(hiddenLayerSize * hiddenLayerSize)
                            .ToList();
        AddLayer(b2, b2.Count, 1, LayerType.Vector);

        // Third Dense Layer
        List<float> d3 = lines[3]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Take(hiddenLayerSize * 1)
                            .ToList();
        AddLayer(d3, hiddenLayerSize, 1, LayerType.Vector);

        // Second Layer Bias
        List<float> b3 = lines[3]
                            .Split(',')
                            .Select(x => float.Parse(x.Trim()))
                            .Skip(1 * hiddenLayerSize)
                            .ToList();
        AddLayer(b3, b3.Count, 1, LayerType.Vector);
        _layersUpdated = true;
    }
}