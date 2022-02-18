package gym;

import engine.core.*;
import engine.helper.Assets;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import py4j.GatewayServer;

import javax.swing.*;
import java.awt.*;
import java.awt.image.VolatileImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class MarioGym {
    String level;
    int gameSeconds;
    int marioState;
    //static boolean visual;

    //Visualisation
    JFrame window = null;
    MarioRender render = null;
    VolatileImage renderTarget = null;
    Graphics backBuffer = null;
    Graphics currentBuffer = null;

    //Game and character
    MarioWorld world = null;
    Py4JAgent agent = null;
    MarioTimer agentTimer = null;

    //GameLoop
    ArrayList<MarioEvent> gameEvents = null;
    ArrayList<MarioAgentEvent> agentEvents = null;

    //Step related
    float rewardPos = 0;
    int rewardTimePenalty = 0;
    int rewardDeathPenalty = 0;

    int winLooseReward = 15;
    int sceneDetail = 0;
    int enemyDetail = 0;

    int totalReward = 0;

    /*
    public static void main(String[] args) {
        MarioGym gym = new MarioGym();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(gym);
        server.start();
        System.out.println("Gateway Started");
    }
    */

    public StepReturnType step(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        agentInput(left, right, down, speed, jump);
        gameUpdate();
        StepReturnType returnVal = new StepReturnType();
        //Done value
        if (world.gameStatus == GameStatus.RUNNING) returnVal.done = false;
        else returnVal.done = true;
        //Reward value
        returnVal.reward = (int) rewardPos + rewardTimePenalty + rewardDeathPenalty;
        returnVal.reward = Math.max(-winLooseReward, Math.min(winLooseReward, returnVal.reward));
        //State value
        returnVal.state = world.getOneHotObservation(world.mario.x, world.mario.y);
        //Info values
        returnVal.info = new HashMap<>();
        if(world.gameStatus == GameStatus.WIN) returnVal.info.put("Result", "Win");
        else if (world.gameStatus == GameStatus.LOSE) returnVal.info.put("Result", "Lose");
        returnVal.info.put("Yolo","Swaggins");
        totalReward += returnVal.reward;
        returnVal.info.put("ReturnScore", String.valueOf(totalReward));
        return returnVal;
    }

    public void init(String paramLevel, String imageDirectory, int timer, int paramMarioState, boolean visual){
        level = paramLevel;
        gameSeconds = timer;
        marioState = paramMarioState;
        Assets.img = imageDirectory;
        sceneDetail = 1;
        enemyDetail = 1;

        if (visual) {
            window = new JFrame("Mario AI Framework");
            render = new MarioRender(2);
            window.setContentPane(render);
            window.pack();
            window.setResizable(false);
            window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            render.init();
            window.setVisible(true);
        }

        reset(visual);
        System.out.println("Gym initialised");
    }

    public void gameUpdate(){
        if (world.gameStatus == GameStatus.RUNNING) {
            //System.out.println(currentTime);
            //get actions
            agentTimer = new MarioTimer(MarioGame.maxTime);
            boolean[] actions = agent.getActions(new MarioForwardModel(world.clone()), agentTimer);
            if (MarioGame.verbose) {
                if (agentTimer.getRemainingTime() < 0 && Math.abs(agentTimer.getRemainingTime()) > MarioGame.graceTime) {
                    System.out.println("The Agent is slowing down the game by: "
                            + Math.abs(agentTimer.getRemainingTime()) + " msec.");
                }
            }
            //Reward info before update
            int tickBeforeUpdate = world.currentTick;
            float marioXBeforeUpdate = world.mario.x;
            // update world
            world.update(actions);
            gameEvents.addAll(world.lastFrameEvents);
            agentEvents.add(new MarioAgentEvent(actions, world.mario.x,
                    world.mario.y, (world.mario.isLarge ? 1 : 0) + (world.mario.isFire ? 1 : 0),
                    world.mario.onGround, world.currentTick));
            //Reward info after update
            int tickAfterUpdate = world.currentTick;
            float marioXAfterUpdate = world.mario.x;
            //Calculate reward components
            rewardPos = marioXAfterUpdate - marioXBeforeUpdate;
            rewardTimePenalty = tickBeforeUpdate - tickAfterUpdate;
            if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
            else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward;
            else rewardDeathPenalty = 0;
            //System.out.println("Postion reward: " + rewardPos + ", Time reward: " + rewardTimePenalty + ", Death reward: " + rewardDeathPenalty);

        }
    }

    public StepReturnType reset(boolean visual){
        agent = new Py4JAgent();
        world = new MarioWorld(null);

        world.visuals = visual;
        world.initializeLevel(level, 1000 * gameSeconds);
        if (visual) {
            world.initializeVisuals(render.getGraphicsConfiguration());
        }
        world.mario.isLarge = marioState > 0;
        world.mario.isFire = marioState > 1;

        world.update(new boolean[MarioActions.numberOfActions()]);

        //initialize graphics
        renderTarget = null;
        backBuffer = null;
        currentBuffer = null;
        if (visual) {
            renderTarget = render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            render.addFocusListener(render); //TODO: Maybe not needed
        }

        agentTimer = new MarioTimer(MarioGame.maxTime);
        agent.initialize(new MarioForwardModel(world.clone()), agentTimer);

        gameEvents = new ArrayList<>();
        agentEvents = new ArrayList<>();

        System.out.println("Gym Reset : Return=" + totalReward);
        totalReward = 0;

        StepReturnType returnVal = new StepReturnType();
        returnVal.done = false;
        returnVal.reward = 0;
        returnVal.state = world.getOneHotObservation(world.mario.x, world.mario.y);
        returnVal.info = new HashMap<>();
        return returnVal;
    }

    public void render(){
        render.renderWorld(world, renderTarget, backBuffer, currentBuffer);
    }

    public void agentInput(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        boolean[] actions = new boolean[]{left, right, down, speed, jump};
        agent.setActions(actions);
    }

    public void setLevel(String levelParam){
        level = levelParam;
    }
}
