/*
Daniel - A01199140
*/

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.IO;
using System.Threading;
using UnityEngine;

[Serializable]
public class AgentData
{
    public int id;
    public float x;
    public float y;
    public float z;
    public float rotation;
}

[Serializable]
public class StateSnapshot
{
    public int step;
    public List<AgentData> agents;
}

public class PythonClient : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 5000;
    public float requestsPerSecond = 10f;
    public GameObject agentCube;

    private TcpClient client;
    private StreamWriter writer;
    private StreamReader reader;
    private Thread networkThread;
    private bool running = true;

    private StateSnapshot latestState;
    private readonly object stateLock = new object();

    void Start()
    {
        networkThread = new Thread(NetworkLoop);
        networkThread.IsBackground = true;
        networkThread.Start();
    }

    void NetworkLoop()
    {
        while (running)
        {
            try
            {
                Connect();
                RequestLoop();
            }
            catch (Exception e)
            {
                Debug.LogWarning("PythonClient: " + e.Message);
            }

            Disconnect();

            if (running)
            {
                Thread.Sleep(1000);
            }
        }
    }

    void Connect()
    {
        client = new TcpClient();
        client.Connect(host, port);
        NetworkStream stream = client.GetStream();
        writer = new StreamWriter(stream);
        writer.AutoFlush = true;
        reader = new StreamReader(stream);
        Debug.Log("PythonClient: conectado a " + host + ":" + port);
    }

    void RequestLoop()
    {
        int delayMs = Mathf.RoundToInt(1000f / requestsPerSecond);

        while (running && client != null && client.Connected)
        {
            writer.WriteLine("GET_STATE");
            string line = reader.ReadLine();

            if (line == null)
            {
                break;
            }

            StateSnapshot snapshot = JsonUtility.FromJson<StateSnapshot>(line);

            lock (stateLock)
            {
                latestState = snapshot;
            }

            Thread.Sleep(delayMs);
        }
    }

    void Disconnect()
    {
        try { reader?.Close(); } catch { }
        try { writer?.Close(); } catch { }
        try { client?.Close(); } catch { }
        reader = null;
        writer = null;
        client = null;
    }

    void Update()
    {
        StateSnapshot snapshot;

        lock (stateLock)
        {
            snapshot = latestState;
        }

        if (snapshot == null || snapshot.agents == null || snapshot.agents.Count == 0)
        {
            return;
        }

        if (agentCube == null)
        {
            return;
        }

        AgentData agent = snapshot.agents[0];
        agentCube.transform.position = new Vector3(agent.x, agent.y, agent.z);
    }

    void OnDestroy()
    {
        running = false;
        Disconnect();
    }
}
